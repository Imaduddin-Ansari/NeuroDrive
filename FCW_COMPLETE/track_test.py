import sys
import cv2
import argparse
import numpy as np
import torch
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

from yolox.tracker.byte_tracker import BYTETracker


def main(video_path):
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # ByteTracker parameters
    args = argparse.Namespace(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30,
        mot20=False
    )

    tracker = BYTETracker(args)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # For now, just pass empty detections (no YOLO yet)
        dummy_detections = torch.from_numpy(np.empty((0, 6))).float()

        # Run tracker update
        online_targets = tracker.update(
            dummy_detections,
            [frame.shape[0], frame.shape[1]],
            (frame.shape[0], frame.shape[1])
        )

        # Draw tracking results (will be empty with dummy detections)
        for target in online_targets:
            tlbr = target.tlbr
            track_id = target.track_id
            cv2.rectangle(frame, (int(tlbr[0]), int(tlbr[1])), (int(tlbr[2]), int(tlbr[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (int(tlbr[0]), int(tlbr[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Overlay info
        cv2.putText(frame, f"Frame: {frame_id}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracks: {len(online_targets)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ByteTrack Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python track_test.py <video_path>")
    else:
        main(sys.argv[1])
