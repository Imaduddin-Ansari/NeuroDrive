"""Traffic Sign Detection processor.

Architecture
────────────
• _video_thread  — reads frames at native FPS, always replaces _detect_queue
                   with the NEWEST frame (evict-and-put pattern, size=1).
• _detect_thread — drains any backlog before running YOLO so it always works
                   on the most-recent frame, never a stale one.
• No bounding boxes or overlays are drawn on the video feed.
• Public attributes (current_sign, sign_confidence) are updated for alerts.
"""
import cv2
import time
import sys
import threading
import queue as _queue
from pathlib import Path

from .base_processor import BaseProcessor
from TrafficSignDetection.TrafficSign import TrafficSignDetector

traffic_module_path = Path(__file__).parent.parent.parent / "TrafficSignDetection"
if str(traffic_module_path) not in sys.path:
    sys.path.insert(0, str(traffic_module_path))


class TrafficSignProcessor(BaseProcessor):
    """Wrapper for traffic sign detection.
    Video display is fully decoupled from YOLO inference — video always plays
    at native FPS while detection runs asynchronously in a separate thread.
    """

    # Keep detections visible for this many seconds after the last positive hit
    # so brief YOLO misses don't clear the alert immediately.
    DETECTION_PERSIST_SECS = 2.5

    def __init__(self, video_source, model_path, class_names_path, alert_threshold=0.80,
                 use_image_completion=True, completion_threshold=0.75, templates_dir=None):
        super().__init__(video_source)

        self.video_sources         = video_source if isinstance(video_source, list) else [video_source]
        self.current_video_index   = -1
        self.cap                   = None
        self.model_path            = model_path
        self.class_names_path      = class_names_path
        self.alert_threshold       = alert_threshold
        self.current_sign          = None
        self.sign_confidence       = 0.0
        self.latest_detections     = []
        self.detector              = None
        self.use_image_completion  = use_image_completion
        self.completion_threshold  = completion_threshold
        self.templates_dir         = templates_dir
        self.inpainter             = None
        self.detection_enabled     = True
        self._video_fps            = 30.0

        # Queue of size 1 — video thread evicts stale frame and puts newest,
        # so detect thread always works on live video with zero backlog.
        self._detect_queue        = _queue.Queue(maxsize=1)
        self._last_detections     = []
        self._detections_lock     = threading.Lock()
        self._last_detection_time = 0.0

        if self.use_image_completion and templates_dir:
            self._initialize_inpainter()

    # ── Inpainter (optional) ──────────────────────────────────────────────────

    def _initialize_inpainter(self):
        try:
            from TrafficSignDetection.ImageCompletion import TrafficSignInpainter
            self.inpainter = TrafficSignInpainter(
                classification_model_path=self.model_path,
                class_names_path=self.class_names_path,
                templates_dir=self.templates_dir,
                inpaint_method='hybrid',
                conservative_mask=True,
                verbose=False
            )
            print("✓ Image completion module initialized")
        except Exception as e:
            print(f"⚠ Image completion unavailable: {e}")
            self.use_image_completion = False

    def _try_image_completion(self, frame, bbox, confidence, predicted_class):
        if not self.use_image_completion or not self.inpainter:
            return confidence, predicted_class, None
        try:
            x1, y1, x2, y2 = map(int, bbox)
            roi = frame[y1:y2, x1:x2].copy()
            sign_class, sign_conf = self.inpainter.classify_sign(roi)
            if sign_conf < self.completion_threshold:
                mask = self.inpainter.generate_occlusion_mask(roi)
                if mask.sum() / (mask.size * 255) > 0.05:
                    improved = self.inpainter.inpaint_hybrid(roi, sign_class, mask)
                    new_class, new_conf = self.inpainter.classify_sign(improved)
                    if new_conf > sign_conf:
                        return new_conf, new_class, improved
            return sign_conf, sign_class, roi
        except Exception:
            return confidence, predicted_class, None

    # ── Video source helpers ──────────────────────────────────────────────────

    def _switch_to_next_video(self):
        if self.cap:
            self.cap.release()
        self.current_video_index = (self.current_video_index + 1) % len(self.video_sources)
        self.cap = cv2.VideoCapture(self.video_sources[self.current_video_index])
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        return self.cap.isOpened()

    # ── Detection thread ──────────────────────────────────────────────────────

    def _detect_thread_fn(self):
        """Always drains the queue to the latest frame before running YOLO —
        no backlog, no stale-frame lag."""
        while self.is_running:
            try:
                frame = self._detect_queue.get(timeout=0.1)
            except _queue.Empty:
                continue

            if frame is None:
                break

            # Drain any extra frames that piled up while we were busy —
            # keep only the newest so we always run on live video.
            while True:
                try:
                    newer = self._detect_queue.get_nowait()
                    if newer is None:
                        return
                    frame = newer
                except _queue.Empty:
                    break

            if not self.detection_enabled or self.detector is None:
                with self._detections_lock:
                    self._last_detections = []
                continue

            try:
                # Single YOLO pass — no augmentation triples
                small = cv2.resize(frame, None, fx=0.5, fy=0.5)
                results = self.detector.yolo_model(small, verbose=False, conf=0.12)
                res = results[0]

                detected_signs = []
                seen = set()

                if res.boxes is not None:
                    for box in res.boxes:
                        cls  = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = self.detector.yolo_model.names[cls]

                        if not any(kw.lower() in name.lower()
                                   for kw in self.detector.traffic_sign_keywords):
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Scale back to original frame coords
                        x1, y1, x2, y2 = x1*2, y1*2, x2*2, y2*2
                        x1 = max(0, min(x1, frame.shape[1] - 1))
                        y1 = max(0, min(y1, frame.shape[0] - 1))
                        x2 = max(0, min(x2, frame.shape[1] - 1))
                        y2 = max(0, min(y2, frame.shape[0] - 1))

                        key = (x1 // 10, y1 // 10, x2 // 10, y2 // 10)
                        if key in seen:
                            continue
                        seen.add(key)

                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        classification = self.detector.classify_sign(crop)
                        if classification and classification['confidence'] >= 0.7:
                            det = {
                                'class':            classification['class'],
                                'confidence':       classification['confidence'],
                                'bbox':             (x1, y1, x2, y2),
                                'yolo_class':       name,
                                'yolo_confidence':  conf,
                            }
                            # Optional image completion
                            if (self.use_image_completion and
                                    det['confidence'] < self.completion_threshold):
                                ic, icls, _ = self._try_image_completion(
                                    frame, det['bbox'],
                                    det['confidence'], det['class'])
                                if ic > det['confidence']:
                                    det['confidence'] = ic
                                    det['class']      = icls
                            detected_signs.append(det)

                with self._detections_lock:
                    if detected_signs:
                        self._last_detections = detected_signs
                        self._last_detection_time = time.time()
                    elif time.time() - self._last_detection_time > self.DETECTION_PERSIST_SECS:
                        # Only clear after the persistence window expires
                        self._last_detections = []

                # Update public attributes (read by main thread for alerts)
                # Only overwrite with "no detection" after the persist window expires
                if detected_signs:
                    self.detection_status = True
                    best = max(detected_signs, key=lambda x: x['confidence'])
                    self.current_sign    = best['class']
                    self.sign_confidence = best['confidence']
                    self.latest_detections = detected_signs
                elif time.time() - self._last_detection_time > self.DETECTION_PERSIST_SECS:
                    self.detection_status  = False
                    self.current_sign      = None
                    self.sign_confidence   = 0.0
                    self.latest_detections = []

            except Exception as e:
                print(f"[TrafficSign] Detect error: {e}")

    # ── Video thread ──────────────────────────────────────────────────────────

    def _video_thread_fn(self):
        """Reads video at native FPS. Pushes clean frames (no boxes) for
        display. For detection, always evicts the stale queued frame and
        replaces it with the current one — zero backlog guaranteed."""
        if not self._switch_to_next_video():
            print(f"✗ Traffic sign: Failed to open {self.video_sources[0]}")
            self.is_running = False
            return

        print(f"✓ Traffic sign video thread @ {self._video_fps:.1f}fps (display capped at 15fps)")
        last_frame_t    = time.time()
        target_interval = 1.0 / 15.0

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                if not self._switch_to_next_video():
                    break
                last_frame_t    = time.time()
                target_interval = 1.0 / 15.0
                continue

            # Evict stale frame, put current — detect thread always gets newest.
            if self.detection_enabled:
                try:
                    self._detect_queue.get_nowait()
                except _queue.Empty:
                    pass
                try:
                    self._detect_queue.put_nowait(frame.copy())
                except _queue.Full:
                    pass

            # Push clean frame (no overlays, no boxes) to display queue
            self._put_frame(frame.copy())

            elapsed    = time.time() - last_frame_t
            sleep_time = target_interval - elapsed
            if sleep_time > 0.002:
                time.sleep(sleep_time)
            last_frame_t = time.time()

        if self.cap:
            self.cap.release()
        try:
            self._detect_queue.put_nowait(None)
        except _queue.Full:
            pass
        print("✓ Traffic sign video thread stopped")

    # ── Entry point ───────────────────────────────────────────────────────────

    def _run(self):
        try:
            print("Initializing TrafficSignDetector...")
            self.detector = TrafficSignDetector(self.model_path, self.class_names_path)
            print("✓ TrafficSignDetector ready")
        except Exception as e:
            print(f"✗ Traffic sign detector init failed: {e}")
            self.detector = None

        # Start the async detection thread first
        dt = threading.Thread(target=self._detect_thread_fn, daemon=True,
                              name="TSDetect")
        dt.start()

        # Run the video thread in this thread (blocks until stopped)
        self._video_thread_fn()

        # Wait for detection thread to finish
        dt.join(timeout=2.0)
        print("✓ Traffic sign processor stopped")