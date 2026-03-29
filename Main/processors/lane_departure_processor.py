"""Lane Departure Warning processor — detection toggle, video always plays"""
import cv2
import sys
import os
import time
import threading
import queue as _queue
from pathlib import Path

from .base_processor import BaseProcessor

lane_module_path = Path(__file__).parent.parent.parent / "LaneDepartureWarning"
scripts_path = lane_module_path / "Scripts"
if scripts_path.exists():
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    lane_module_path = scripts_path
else:
    if str(lane_module_path) not in sys.path:
        sys.path.insert(0, str(lane_module_path))


class LaneDepartureProcessor(BaseProcessor):
    """Handles lane departure warning detection.
    Video always plays; detection toggled via self.detection_enabled.
    """

    def __init__(self, video_source):
        super().__init__(video_source)
        self.lane_detector     = None
        self.direction         = 'F'
        self.curvature         = 0.0
        self.position          = 0.0
        self.deviation_warning = False
        self.warning_message   = None
        self.detection_failed  = False
        self.left_indicator    = False
        self.right_indicator   = False
        self.detection_enabled = True

        # Queue size=1 — evict-and-put keeps detect thread on newest frame
        self._detect_queue = _queue.Queue(maxsize=1)
        self._lane_lock    = threading.Lock()
        self._lane_data    = {}   # latest results from detect thread

    def set_indicators(self, left=False, right=False):
        self.left_indicator  = left
        self.right_indicator = right
        if self.lane_detector:
            self.lane_detector.set_indicator(left=left, right=right)

    def _detect_thread_fn(self):
        """Runs lane detection on the newest frame — drains queue backlog first."""
        while self.is_running:
            try:
                frame = self._detect_queue.get(timeout=0.1)
            except _queue.Empty:
                continue
            if frame is None:
                break

            # Drain to newest
            while True:
                try:
                    newer = self._detect_queue.get_nowait()
                    if newer is None:
                        return
                    frame = newer
                except _queue.Empty:
                    break

            if not self.detection_enabled or self.lane_detector is None:
                with self._lane_lock:
                    self._lane_data = {}
                continue
            try:
                _, data = self.lane_detector.process_frame(frame)
                with self._lane_lock:
                    self._lane_data = data
                self.direction         = data.get('direction', 'F')
                self.curvature         = data.get('curvature', 0.0)
                self.position          = data.get('position', 0.0)
                self.deviation_warning = data.get('deviation_warning', False)
                self.warning_message   = data.get('warning_message', None)
                self.detection_failed  = data.get('detection_failed', False)
                self.detection_status  = self.deviation_warning
            except Exception as e:
                print(f"[Lane] Detect error: {e}")

    def _run(self):
        try:
            try:
                import docopt
            except ImportError:
                print("✗ Lane Departure requires 'docopt' — pip install docopt")
                self.is_running = False
                return

            import importlib.util

            lane_main_candidates = [
                lane_module_path / "main.py",
                lane_module_path.parent / "main.py",
            ]
            lane_main_path = next((p for p in lane_main_candidates if p.exists()), None)
            if lane_main_path is None:
                print("✗ Lane Departure: main.py not found in LaneDepartureWarning")
                self.is_running = False
                return

            spec = importlib.util.spec_from_file_location("lane_main", str(lane_main_path))
            lane_main = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lane_main)
            FindLaneLines = lane_main.FindLaneLines

            cal_path = lane_module_path.parent / "Images" / "camera_cal"
            if not cal_path.exists():
                cal_path = lane_module_path / "Images" / "camera_cal"

            original_cwd = os.getcwd()
            lane_base    = (lane_module_path.parent
                            if lane_module_path.name == 'Scripts'
                            else lane_module_path)
            os.chdir(str(lane_base))
            try:
                self.lane_detector = FindLaneLines(
                    camera_cal_path=str(cal_path),
                    show_overlay=False,
                    img_size=(1280, 720)
                )
            finally:
                os.chdir(original_cwd)

            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                print(f"✗ Lane: Failed to open {self.video_source}")
                self.is_running = False
                return

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            video_fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
            target_interval = 1.0 / 15.0  # cap at 15fps so detection stays in sync
            print(f"✓ Lane Departure Warning started @ {video_fps:.1f}fps (display capped at 15fps)")

            # Start async detection thread
            dt = threading.Thread(target=self._detect_thread_fn,
                                  daemon=True, name="LaneDetect")
            dt.start()

            frame_count  = 0
            last_frame_t = time.time()

            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    last_frame_t = time.time()
                    continue

                frame_count += 1

                # Evict stale frame, put current — zero backlog
                try:
                    self._detect_queue.get_nowait()
                except _queue.Empty:
                    pass
                try:
                    self._detect_queue.put_nowait(frame.copy())
                except _queue.Full:
                    pass

                if not self.detection_enabled:
                    self.deviation_warning = False
                    self.warning_message   = None
                    self.detection_status  = False

                # Always put raw frame for display
                self._put_frame(frame.copy())

                elapsed    = time.time() - last_frame_t
                sleep_time = target_interval - elapsed
                if sleep_time > 0.002:
                    time.sleep(sleep_time)
                last_frame_t = time.time()

            cap.release()
            try:
                self._detect_queue.put_nowait(None)
            except _queue.Full:
                pass
            dt.join(timeout=2.0)
            print("✓ Lane Departure Warning stopped")

        except ImportError as e:
            print(f"✗ Lane Departure import error: {e}")
            self.is_running = False
        except Exception as e:
            print(f"✗ Lane Departure error: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False