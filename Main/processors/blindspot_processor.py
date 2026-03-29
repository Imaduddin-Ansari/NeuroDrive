"""Blind Spot Monitoring processor — SERIOUS rapid 3-beep alert + video always plays"""
import cv2
import os
import sys
import time
import threading

from .base_processor import BaseProcessor

# ── Audio alert — SERIOUS: rapid 3 short high beeps (parking radar urgency) ──
_bsp_beep_lock = threading.Lock()

try:
    import winsound as _winsound
    def _bsp_play():
        _winsound.Beep(1900, 150)
        time.sleep(0.06)
        _winsound.Beep(1900, 150)
        time.sleep(0.06)
        _winsound.Beep(1900, 150)
except ImportError:
    def _bsp_play():
        if sys.platform == 'darwin':
            for _ in range(3):
                os.system('afplay /System/Library/Sounds/Tink.aiff 2>/dev/null &')
                time.sleep(0.18)
        else:
            os.system(
                'beep -f 1900 -l 150 -D 60 -f 1900 -l 150 -D 60 -f 1900 -l 150 2>/dev/null || '
                '(printf "\\a"; sleep 0.15; printf "\\a"; sleep 0.15; printf "\\a")'
            )

def _bsp_beep_async():
    if not _bsp_beep_lock.acquire(blocking=False):
        return
    def _run():
        try:
            _bsp_play()
        finally:
            _bsp_beep_lock.release()
    threading.Thread(target=_run, daemon=True).start()


class BlindSpotProcessor(BaseProcessor):
    """Handles blind spot monitoring for one side (left or right).
    Video always plays; detection toggled via self.detection_enabled.
    """

    AUDIO_COOLDOWN = 4.0

    def __init__(self, video_source, side='left'):
        super().__init__(video_source)
        self.side              = side
        self.vehicle_count     = 0
        self.vehicle_distance  = None
        self.vehicle_details   = []
        self.monitor           = None
        self._last_beep_time   = 0.0
        self.detection_enabled = True

    def _run(self):
        try:
            from BlindSpotMonitoring.SourceCode.blindspot import BlindSpotMonitor
            self.monitor = BlindSpotMonitor(
                side=self.side,
                model_name="midas_v21_small",
                models_dir="../BlindSpotMonitoring/SourceCode/Model",
                detection_interval=5,
                verbose=False,
            )
        except Exception as e:
            print(f"[BlindSpot-{self.side}] Monitor load failed: {e}. Video-only mode.")
            self.monitor = None

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"[BlindSpot-{self.side}] Failed to open: {self.video_source}")
            self.is_running = False
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Cap the display+detection loop at 15 fps.
        # The side cameras don't need more than 15fps; running flat-out
        # wastes a full CPU core per camera on a MacBook M-series chip.
        TARGET_FPS = 15.0
        target_interval = 1.0 / TARGET_FPS
        # Run depth/YOLO inference on every Nth frame to save CPU.
        # At 15 fps display, inferring every 3rd frame = 5 Hz detection —
        # fast enough for a vehicle closing at highway speed.
        DETECT_EVERY = 3
        frame_idx = 0
        last_frame_t = time.time()
        print(f"[BlindSpot] Started — {self.side} side (display {TARGET_FPS:.0f}fps, detect every {DETECT_EVERY} frames)")

        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                last_frame_t = time.time()
                continue

            frame_idx += 1
            run_detection = (frame_idx % DETECT_EVERY == 0)

            if self.detection_enabled and self.monitor is not None and run_detection:
                try:
                    processed_frame, vehicles = self.monitor.process_frame(frame, draw_overlay=False)
                except Exception:
                    processed_frame, vehicles = frame, []

                self.detection_status = len(vehicles) > 0
                self.vehicle_count    = len(vehicles)
                self.vehicle_details  = vehicles

                if vehicles:
                    with_dist = [v for v in vehicles if v.get('distance')]
                    self.vehicle_distance = (
                        min(with_dist, key=lambda v: v['distance'])['distance']
                        if with_dist else None
                    )
                    now = time.time()
                    if now - self._last_beep_time >= self.AUDIO_COOLDOWN:
                        self._last_beep_time = now
                        _bsp_beep_async()
                else:
                    self.vehicle_distance = None
            else:
                if not (self.detection_enabled and self.monitor is not None):
                    self.detection_status = False
                    self.vehicle_count    = 0
                    self.vehicle_distance = None
                processed_frame = frame

            self._put_frame(processed_frame)

            # Pace to TARGET_FPS — sleep for remaining time in this frame slot.
            elapsed = time.time() - last_frame_t
            sleep_t = target_interval - elapsed
            if sleep_t > 0.001:
                time.sleep(sleep_t)
            last_frame_t = time.time()

        cap.release()
        print(f"[BlindSpot] Stopped — {self.side} side")