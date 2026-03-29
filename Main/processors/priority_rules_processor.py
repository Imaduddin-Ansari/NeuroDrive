"""Priority-Based Rules Detection processor — detection toggle, video always plays"""
import cv2
import sys
import time
import threading
import queue as _queue
from pathlib import Path

from .base_processor import BaseProcessor

priority_module_path = Path(__file__).parent.parent.parent / "PriorityRulesAlert"
if str(priority_module_path) not in sys.path:
    sys.path.insert(0, str(priority_module_path))


class PriorityRulesProcessor(BaseProcessor):
    """Priority-based traffic rules detection.
    Video always plays at native FPS. Detection always runs on the newest
    frame — queue size 1, evict-and-put pattern, no backlog, no latency.
    No boxes or overlays are drawn on the video feed.
    """

    DETECTION_PERSIST_SECS = 2.5

    def __init__(self, video_source):
        super().__init__(video_source)
        self.detector             = None
        self.active_rules         = []
        self.critical_rules       = []
        self.high_priority_rules  = []
        self.original_width       = None
        self.original_height      = None
        self.detection_enabled    = True

        self._last_rules          = []
        self._rules_lock          = threading.Lock()
        # Size=1: video thread always evicts stale frame and puts newest
        self._detect_queue        = _queue.Queue(maxsize=1)
        self._last_detection_time = 0.0

    # ── Detection thread ──────────────────────────────────────────────────────

    def _detect_thread_fn(self):
        while self.is_running:
            try:
                frame = self._detect_queue.get(timeout=0.1)
            except _queue.Empty:
                continue
            if frame is None:
                break

            # Drain any extra frames — always work on the newest
            while True:
                try:
                    newer = self._detect_queue.get_nowait()
                    if newer is None:
                        return
                    frame = newer
                except _queue.Empty:
                    break

            if not self.detection_enabled or self.detector is None:
                with self._rules_lock:
                    self._last_rules = []
                continue
            try:
                _, detections, triggered_rules = self.detector.process_frame(
                    frame, show_rules=False)
                now = time.time()
                with self._rules_lock:
                    if triggered_rules:
                        self._last_rules          = triggered_rules
                        self._last_detection_time = now
                    elif now - self._last_detection_time > self.DETECTION_PERSIST_SECS:
                        self._last_rules = []
                self.active_rules        = self._last_rules
                self.critical_rules      = [r for r in self._last_rules if r.get('priority') == 'critical']
                self.high_priority_rules = [r for r in self._last_rules if r.get('priority') == 'high']
                self.detection_status    = bool(self.critical_rules or self.high_priority_rules)
            except Exception as e:
                print(f"[Priority Rules] Detect error: {e}")

    # ── Main run ──────────────────────────────────────────────────────────────

    def _run(self):
        try:
            print("[Priority Rules] Attempting to initialize...")
            try:
                from PriorityRulesDetecting import TrafficRuleDetector
                print("[Priority Rules] ✓ Imported TrafficRuleDetector")
            except ImportError as ie:
                print(f"[Priority Rules] ✗ Import error: {ie}")
                self.detector = None
            else:
                kb_path = priority_module_path / "traffic_rules_knowledge_base.json"
                if kb_path.exists():
                    self.detector = TrafficRuleDetector(knowledge_base_path=str(kb_path))
                    print("[Priority Rules] ✓ Detector initialized")
                else:
                    print(f"[Priority Rules] ✗ Knowledge base not found: {kb_path}")
                    self.detector = None

            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                print(f"[Priority Rules] ✗ Failed to open video: {self.video_source}")
                self.is_running = False
                return

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.original_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            target_interval = 1.0 / 15.0  # cap display at 15fps so detection stays in sync
            print(f"[Priority Rules] Video {self.original_width}x{self.original_height} @ {video_fps:.1f}fps (display capped at 15fps)")
            print("✓ Priority Rules processor started")

            dt = threading.Thread(target=self._detect_thread_fn, daemon=True,
                                  name="PRDetect")
            dt.start()

            last_frame_t = time.time()

            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    last_frame_t = time.time()
                    continue

                # Evict stale frame, put current — no backlog
                if self.detection_enabled:
                    try:
                        self._detect_queue.get_nowait()
                    except _queue.Empty:
                        pass
                    try:
                        self._detect_queue.put_nowait(frame.copy())
                    except _queue.Full:
                        pass

                if not self.detection_enabled:
                    self.active_rules        = []
                    self.critical_rules      = []
                    self.high_priority_rules = []
                    self.detection_status    = False

                # Push clean frame (no boxes) to display queue
                display_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                self._put_frame(display_frame)

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
            print("✓ Priority Rules processor stopped")

        except Exception as e:
            print(f"✗ Priority Rules error: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False