#!/usr/bin/env python3
"""
fcw_pipeline.py - YOLOv8n + ByteTrack + (MiDaS optional) FCW prototype
 - If MiDaS + weights are present, uses MiDaS for depth.
 - Otherwise uses a bbox-height-based depth proxy (immediate fallback).
Usage:
    python fcw_pipeline.py ./samplevideos/test.mov [optional_yolo_weights]
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
import cv2

# Fix deprecated numpy aliases some code expects
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# Ensure ByteTrack repo is importable when running in its root
# (This file should be placed in the ByteTrack repo root.)
if os.path.isdir(os.path.join(os.path.dirname(__file__), "yolox")):
    sys.path.insert(0, os.path.dirname(__file__))

# ByteTrack
try:
    from yolox.tracker.byte_tracker import BYTETracker
except Exception as e:
    print("[ERROR] Could not import BYTETracker from yolox. Make sure you run this from ByteTrack root.")
    raise

# YOLOv8
from ultralytics import YOLO

# Try to import MiDaS (optional). If not available, we will fallback to bbox-proxy depth.
MIDAS_AVAILABLE = False
midas_model = None
midas_transform = None
try:
    # If you cloned MiDaS into ByteTrack/MiDaS, add to path
    midas_dir = os.path.join(os.path.dirname(__file__), "MiDaS")
    if os.path.isdir(midas_dir) and midas_dir not in sys.path:
        sys.path.insert(0, midas_dir)
    from midas.model_loader import load_model as load_midas_model
    MIDAS_AVAILABLE = True
except Exception:
    MIDAS_AVAILABLE = False

# Helper: try to initialize MiDaS if available and weights exist
def try_init_midas(device):
    global midas_model, midas_transform, MIDAS_AVAILABLE
    if not MIDAS_AVAILABLE:
        return False
    # Common MiDaS small weights filename (what users typically download)
    weights_rel = os.path.join("MiDaS", "weights", "dpt_small-midas-2f21e586.pt")
    if not os.path.exists(weights_rel):
        print("[MI DAS] weights not found at:", weights_rel)
        print("[MI DAS] If you want MiDaS, download weights to MiDaS/weights/dpt_small-midas-2f21e586.pt")
        return False
    try:
        # The MiDaS loader varies across commits. We'll attempt common signatures.
        # Preferred call: load_midas_model(device, model_type, weights_path) OR load_midas_model(device, weights_path)
        try:
            midas, transform, net_w, net_h, resize_mode, normalization = load_midas_model(device, "dpt_small", weights_rel)
        except TypeError:
            # fallback: loader may expect (device, weights_path)
            midas, transform, net_w, net_h, resize_mode, normalization = load_midas_model(device, weights_rel)
        midas_model = midas
        midas_transform = transform
        print("[MiDaS] initialized using weights:", weights_rel)
        return True
    except Exception as e:
        print("[MiDaS] failed to init:", e)
        MIDAS_AVAILABLE = False
        return False

def estimate_depth_midas(frame, device):
    """Return depth_map (numpy) using midas_model/midas_transform (assumes initialized)."""
    global midas_model, midas_transform
    if midas_model is None or midas_transform is None:
        raise RuntimeError("MiDaS not initialized")
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    data = midas_transform({"image": img})
    img_input = data["image"].to(device).unsqueeze(0)
    with torch.no_grad():
        prediction = midas_model.forward(img_input)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map

# Fallback depth proxy: use bbox height as inverse distance proxy
def depth_proxy_from_bbox_height(bbox, image_shape):
    """
    bbox: (x1,y1,x2,y2)
    image_shape: (h, w)
    Returns numeric proxy (smaller => closer). This is relative, not meters.
    We'll use: proxy = 1.0 / (h_box / img_h + eps)
    """
    x1, y1, x2, y2 = bbox
    img_h = image_shape[0]
    h_box = max(1.0, (y2 - y1))
    frac = h_box / float(img_h)
    proxy = 1.0 / (frac + 1e-6)
    # scale to a comfortable range
    return float(proxy)

# Convert YOLO boxes -> ByteTrack detection tensor [x1,y1,x2,y2,score,cls]
def yolo_to_bytetrack_dets(yolo_boxes, scores, classes):
    if len(yolo_boxes) == 0:
        return torch.from_numpy(np.empty((0,6))).float()
    arr = np.zeros((len(yolo_boxes), 6), dtype=np.float32)
    for i, box in enumerate(yolo_boxes):
        x1, y1, x2, y2 = box
        arr[i,0] = x1
        arr[i,1] = y1
        arr[i,2] = x2
        arr[i,3] = y2
        arr[i,4] = float(scores[i])
        arr[i,5] = float(classes[i])
    return torch.from_numpy(arr).float()

# Main
def main(video_path, yolo_weights="yolov8n.pt", conf_thresh=0.25, device_str=None):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[INFO] device:", device)

    # load YOLOv8 model
    print("[INFO] loading YOLOv8 weights:", yolo_weights)
    yolo = YOLO(yolo_weights)

    # try MiDaS init (optional)
    midas_ready = try_init_midas(device)

    # prepare ByteTracker
    args = argparse.Namespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30, mot20=False)
    tracker = BYTETracker(args)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] cannot open video:", video_path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print("[INFO] video FPS:", fps)

    frame_id = 0
    start_time = time.time()
    vehicle_classes = {2,3,5,7}  # COCO vehicle classes: car,motorcycle,bus,truck

    # track history for simple velocity estimation (pixel displacement)
    track_prev_centers = {}  # track_id -> (cx, cy)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # YOLO detection
        results = yolo(frame, imgsz=640, conf=conf_thresh)
        res = results[0]

        yolo_boxes = []
        yolo_scores = []
        yolo_classes = []

        if hasattr(res, "boxes") and len(res.boxes) > 0:
            for box in res.boxes:
                cls = int(box.cls.cpu().numpy()) if hasattr(box.cls, "cpu") else int(box.cls)
                conf = float(box.conf.cpu().numpy()) if hasattr(box.conf, "cpu") else float(box.conf)
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, "cpu") else np.array(box.xyxy[0])
                x1, y1, x2, y2 = map(float, xyxy)
                if cls in vehicle_classes:
                    yolo_boxes.append((x1,y1,x2,y2))
                    yolo_scores.append(conf)
                    yolo_classes.append(cls)

        dets_tensor = yolo_to_bytetrack_dets(yolo_boxes, yolo_scores, yolo_classes)

        # Depth map (MiDaS) if available
        depth_map = None
        if midas_ready:
            try:
                depth_map = estimate_depth_midas(frame, device)
            except Exception as e:
                # disable MiDaS if anything goes wrong (robust fallback)
                print("[MiDaS] error during inference, disabling MiDaS:", e)
                midas_ready = False
                depth_map = None

        # Update tracker
        try:
            online_targets = tracker.update(dets_tensor, [frame.shape[0], frame.shape[1]], (frame.shape[0], frame.shape[1]))
        except Exception:
            # fallback signature
            online_targets = tracker.update(dets_tensor, (frame.shape[0], frame.shape[1]), (frame.shape[0], frame.shape[1]))

        # Build info per track
        track_infos = []
        for t in online_targets:
            tlbr = t.tlbr
            track_id = t.track_id
            x1, y1, x2, y2 = map(int, tlbr)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)

            # depth estimate
            if (depth_map is not None) and (0 <= y1 < y2 <= depth_map.shape[0]) and (0 <= x1 < x2 <= depth_map.shape[1]):
                crop = depth_map[y1:y2, x1:x2]
                if crop.size == 0:
                    depth_val = float("inf")
                else:
                    depth_val = float(np.median(crop))
                depth_source = "MiDaS"
            else:
                # fallback proxy
                depth_val = depth_proxy_from_bbox_height((x1,y1,x2,y2), frame.shape[:2])
                depth_source = "proxy"

            # simple pixel-speed estimation (center displacement)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            vel_px = 0.0
            if track_id in track_prev_centers:
                prev_cx, prev_cy = track_prev_centers[track_id]
                # pixels per frame
                vel_px = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
            track_prev_centers[track_id] = (cx, cy)

            track_infos.append({
                "id": track_id,
                "bbox": (x1,y1,x2,y2),
                "depth": depth_val,
                "depth_source": depth_source,
                "vel_px": vel_px
            })

            # draw bbox and label
            label = f"ID:{track_id} {depth_source}:{depth_val:.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
            cv2.putText(frame, label, (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 2)

        # Prototype TTC-like alert using proxy: if depth (proxy) below threshold -> alert
        # (When MiDaS used, depth units are arbitrary; thresholding is empirical.)
        TTC_PROXY_THRESHOLD = 1.6  # tuned for proxy: lower values => closer -> alert; adjust as needed
        for info in track_infos:
            if info["depth"] < TTC_PROXY_THRESHOLD:
                print(f"[ALERT] Frame {frame_id} - Track {info['id']} CLOSE (depth={info['depth']:.3f}, src={info['depth_source']})")

        # overlays
        cv2.putText(frame, f"Frame:{frame_id}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(frame, f"Tracks:{len(online_targets)}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        if midas_ready:
            cv2.putText(frame, "Depth: MiDaS", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(frame, "Depth: proxy (bbox height)", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("FCW Pipeline", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"[INFO] processed frames:{frame_id} time:{elapsed:.2f}s FPS:{frame_id/elapsed:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fcw_pipeline.py <video_path> [yolo_weights]")
        sys.exit(1)
    video_path = sys.argv[1]
    weights = sys.argv[2] if len(sys.argv) > 2 else "yolov8n.pt"
    main(video_path, yolo_weights=weights)
