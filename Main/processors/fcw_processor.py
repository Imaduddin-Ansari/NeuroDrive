"""Forward Collision Warning processor — detection toggle, video always plays"""
import cv2
import time
import os
import sys
import threading
import queue as _queue
import torch
from ultralytics import YOLO

from .base_processor import BaseProcessor
from detection.lane_detector import RobustLaneDetector
from detection.depth_estimator import AccurateDepthEstimator
from detection.ttc_calculator import TTCCalculator
from FCW_COMPLETE.yolox.tracker.byte_tracker import BYTETracker

# ── Audio alert — 3 descending tones (2000→1600→1200 Hz) ─────────────────────
_fcw_beep_lock = threading.Lock()

try:
    import winsound as _winsound
    def _fcw_play():
        _winsound.Beep(2000, 200)
        time.sleep(0.05)
        _winsound.Beep(1600, 200)
        time.sleep(0.05)
        _winsound.Beep(1200, 300)
except ImportError:
    def _fcw_play():
        if sys.platform == 'darwin':
            os.system('afplay /System/Library/Sounds/Basso.aiff -rate 1.5 2>/dev/null')
            time.sleep(0.1)
            os.system('afplay /System/Library/Sounds/Basso.aiff -rate 1.2 2>/dev/null')
            time.sleep(0.1)
            os.system('afplay /System/Library/Sounds/Basso.aiff -rate 0.9 2>/dev/null')
        else:
            os.system(
                'beep -f 2000 -l 200 -D 50 -f 1600 -l 200 -D 50 -f 1200 -l 300 2>/dev/null || '
                '(printf "\\a"; sleep 0.1; printf "\\a"; sleep 0.1; printf "\\a")'
            )

def _fcw_beep_async():
    if not _fcw_beep_lock.acquire(blocking=False):
        return
    def _run():
        try:
            _fcw_play()
        finally:
            _fcw_beep_lock.release()
    threading.Thread(target=_run, daemon=True).start()


class ForwardCollisionProcessor(BaseProcessor):
    """Runs FCW on a video feed. Video always plays; detection toggled via detection_enabled."""

    AUDIO_COOLDOWN = 2.0

    def __init__(self, video_path, yolo_weights="yolov8n.pt", ego_kmph=50.0):
        super().__init__(video_path)
        self.yolo_w            = yolo_weights
        self.ego_kmph          = ego_kmph
        self.critical          = False
        self.detection_enabled = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo   = None
        self.lane   = RobustLaneDetector()
        self.depth  = AccurateDepthEstimator()
        self.ttc    = TTCCalculator()

        args = type('A', (), {
            'track_thresh': 0.5, 'track_buffer': 30,
            'match_thresh': 0.8, 'frame_rate': 30, 'mot20': False,
        })()
        self.tracker = BYTETracker(args)

        self.classes   = {2, 3, 5, 7}
        self.frame_cnt = 0
        self._last_beep_time = 0.0

        # Async detection support
        # Size=1: video thread evicts stale item and puts newest — no backlog
        self._detect_queue  = _queue.Queue(maxsize=1)
        self._overlay_lock  = threading.Lock()
        self._last_overlay  = None
        self._last_alert_time = 0.0

    # ── Detection thread ──────────────────────────────────────────────────────

    def _detect_thread_fn(self, ego_mps, danger_m, ttc_thr):
        """Runs YOLO + tracking. Always drains to the newest frame first
        so there is never a lag between what's on screen and what's detected."""
        import queue as _q
        while self.is_running:
            try:
                item = self._detect_queue.get(timeout=0.1)
            except _q.Empty:
                continue
            if item is None:
                break

            # Drain — keep only newest
            while True:
                try:
                    newer = self._detect_queue.get_nowait()
                    if newer is None:
                        return
                    item = newer
                except _q.Empty:
                    break

            frame, t = item
            if not self.detection_enabled or self.yolo is None:
                with self._overlay_lock:
                    self._last_overlay = None
                self.critical = False
                self.detection_status = False
                continue

            try:
                h, w = frame.shape[:2]
                self.lane.detect_lanes(frame)
                res = self.yolo(frame, imgsz=640, conf=0.35,
                                device=self.device, verbose=False)
                dets = []
                if res and len(res[0].boxes):
                    for b in res[0].boxes:
                        cls = int(b.cls[0])
                        if cls not in self.classes:
                            continue
                        xyxy = b.xyxy[0].cpu().numpy()
                        dets.append({'bbox': list(map(float, xyxy)),
                                     'conf': float(b.conf[0]), 'cls': cls})

                tensor = torch.tensor([
                    [d['bbox'][0], d['bbox'][1], d['bbox'][2],
                     d['bbox'][3], d['conf'], d['cls']] for d in dets
                ]) if dets else torch.empty((0, 6))

                tracks = self.tracker.update(tensor, [h, w], (h, w)) if len(tensor) else []

                overlay = frame.copy()
                critical = False
                for tr in tracks:
                    if not hasattr(tr, 'tlbr'):
                        continue
                    tid = int(tr.track_id)
                    x1, y1, x2, y2 = map(int, tr.tlbr)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    bbox = (x1, y1, x2, y2)

                    in_lane = self.lane.is_in_ego_lane(bbox, frame.shape)
                    depth   = self.depth.estimate_depth(bbox, frame.shape)
                    self.ttc.update(tid, depth, t)
                    ttc_val, rel_v = self.ttc.ttc(tid, ego_mps)

                    is_crit = in_lane and depth < danger_m and ttc_val < ttc_thr and rel_v < -0.5
                    is_warn = in_lane and depth < danger_m * 1.5 and ttc_val < ttc_thr * 2

                    color = ((0, 0, 255) if is_crit else
                             (0, 165, 255) if is_warn else
                             (0, 255, 0)  if in_lane else (128, 128, 128))
                    thick = 4 if is_crit else 3 if is_warn else 2
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thick)
                    if is_crit:
                        cv2.putText(overlay, f"ID:{tid} {depth:.1f}m TTC:{ttc_val:.1f}s",
                                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        critical = True

                if critical:
                    cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 200), -1)
                    cv2.putText(overlay, "!!! COLLISION WARNING !!!",
                                (w // 2 - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
                    cv2.putText(overlay, "BRAKE IMMEDIATELY!",
                                (w // 2 - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    now = time.time()
                    if now - self._last_beep_time >= self.AUDIO_COOLDOWN:
                        self._last_beep_time = now
                        _fcw_beep_async()

                self.critical         = critical
                self.detection_status = critical
                with self._overlay_lock:
                    self._last_overlay = overlay

            except Exception as e:
                print(f"[FCW] Detect error: {e}")

    def _run(self):
        import queue as _q
        try:
            self.yolo = YOLO(self.yolo_w)

            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                print("[FCW] Cannot open video")
                self.is_running = False
                return

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            ego_mps   = self.ego_kmph / 3.6
            danger_m  = 10.0
            ttc_thr   = 3.0
            target_interval = 1.0 / 15.0  # cap at 15fps so detection stays in sync
            print(f"[FCW] Started @ {video_fps:.1f}fps (display capped at 15fps)")

            # Async detection thread
            dt = threading.Thread(
                target=self._detect_thread_fn,
                args=(ego_mps, danger_m, ttc_thr),
                daemon=True, name="FCWDetect")
            dt.start()

            last_frame_t = time.time()
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    last_frame_t = time.time()
                    continue

                self.frame_cnt += 1

                # Evict stale item, put current — zero backlog
                if self.detection_enabled:
                    try:
                        self._detect_queue.get_nowait()
                    except _q.Empty:
                        pass
                    try:
                        self._detect_queue.put_nowait((frame.copy(), time.time()))
                    except _q.Full:
                        pass

                # Display clean frame (no boxes drawn here)
                self._put_frame(frame.copy())

                elapsed    = time.time() - last_frame_t
                sleep_time = target_interval - elapsed
                if sleep_time > 0.002:
                    time.sleep(sleep_time)
                last_frame_t = time.time()

            cap.release()
            try:
                self._detect_queue.put_nowait(None)
            except _q.Full:
                pass
            dt.join(timeout=2.0)
            print("[FCW] Stopped.")

        except Exception as e:
            print(f"[FCW] Error: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False

    def get_frame(self):
        return self.get_processed_frame()

    @property
    def running(self):
        return self.is_running