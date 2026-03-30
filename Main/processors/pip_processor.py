"""
Pedestrian Intent Prediction processor — Module 11

Redesigned as a frame-overlay processor:
  • No separate video file — receives frames from the active front camera
  • push_frame(frame) is called by the main tick loop with whatever is
    currently showing on the front camera (any segment)
  • Processes asynchronously in a background thread so it never blocks the UI
  • get_annotated_frame() returns the latest PIP-annotated frame
  • Works on ALL front camera footage regardless of active segment
"""
import cv2
import time
import threading
import queue
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_PIP_DIR = Path(__file__).parent.parent.parent / "PedestrianIntentPrediction"
sys.path.insert(0, str(_PIP_DIR))

from PIP_done import CompletePedestrianIntentSystem, Config as PIPConfig
# Lightweight env adaptation (from enhanced_pip - minimal)
class WeatherDetector:
    def detect_weather_conditions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features = {'brightness': np.mean(gray)/255, 'contrast': np.std(gray)/255}
        if features['brightness'] < 0.3: return 'night', 0.8
        return 'clear', 0.9
    def get_weather_adjustments(self): return {'detection_threshold': 0.85 if self.current_weather=='clear' else 0.7}

class LightingClassifier:
    def classify_lighting(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)/255
        return 'night' if brightness<0.4 else 'day', brightness
    def get_lighting_adjustments(self): return {'detection_threshold': 0.75 if self.current_lighting=='night' else 0.95}


class PedestrianIntentProcessor:
    """
    Frame-overlay processor for Module 11.

    Usage from main_window tick loop:
        # Push the current front-camera frame in
        pip_processor.push_frame(front_frame)
        # Get back the PIP-annotated version
        annotated = pip_processor.get_annotated_frame()
        if annotated is not None:
            video_grid.update_feed(FEED_FRONT_CAMERA, annotated)
    """

    def __init__(self, model_path=None, vehicle_speed_kmh: float = 30.0):
        self.vehicle_speed_kmh = vehicle_speed_kmh
        self.detection_enabled = True
        self.is_running        = False

        self._model_path = (
            Path(model_path) if model_path
            else PIPConfig.INTENT_MODEL
        )

        # Input queue: main tick pushes raw frames here (size=1, always newest)
        self._in_queue  = queue.Queue(maxsize=1)
        # Output queue: worker pushes annotated frames here (size=1)
        self._out_queue = queue.Queue(maxsize=1)

        self._thread = None
        self._system = None

        # Public state
        self.detection_status = False
        self.crossing_alerts: list = []
        self.statistics: dict = {}
        self._last_annotated = None
        self._last_annotated_lock = threading.Lock()

        # Optimization: batch weather/lighting checks
        self._weather_frame_skip = 5  # Only check every N frames
        self._weather_counter = 0
        self._last_weather_type = 'clear'
        self._last_lighting_type = 'day'
        self._last_weather_adj = 0.9
        self._last_lighting_adj = 0.95
        self._last_dynamic_conf = 0.65
        self._print_counter = 0
        self._print_skip = 5  # Only print debug every N frames

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if not self.is_running:
            self.is_running = True
            self._thread = threading.Thread(
                target=self._worker, daemon=True, name="PIPWorker")
            self._thread.start()

    def stop(self):
        self.is_running = False
        # Unblock the worker
        try:
            self._in_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=3)
        print("[PIP] Stopped.")

    # ── Frame I/O (called from main tick — non-blocking) ──────────────────────

    def push_frame(self, frame):
        """Push a raw front-camera frame for PIP processing. Drops stale frames."""
        if frame is None or not self.is_running:
            return
        # Evict stale frame, put newest
        try:
            self._in_queue.get_nowait()
        except queue.Empty:
            pass
        # Only copy if needed (worker will copy before drawing)
        try:
            self._in_queue.put_nowait(frame)
        except queue.Full:
            pass

    def get_annotated_frame(self):
        """Return the latest PIP-annotated frame, or the last cached one."""
        try:
            frame = self._out_queue.get_nowait()
            with self._last_annotated_lock:
                self._last_annotated = frame
            return frame
        except queue.Empty:
            with self._last_annotated_lock:
                return self._last_annotated

    # ── Background worker ─────────────────────────────────────────────────────


    def _worker(self):
        try:
            print("[PIP] Initializing pedestrian intent system...")
            self._system = CompletePedestrianIntentSystem(
                model_path=self._model_path)
            self._system.alert_system.update_vehicle_speed(
                self.vehicle_speed_kmh)

            # Disable MiDaS — pinhole fallback is 10x faster and sufficient
            if hasattr(self._system, 'depth_estimator'):
                self._system.depth_estimator._midas = None

            # Prepare detectors for dynamic confidence
            if hasattr(self._system, 'detector'):
                self._weather_detector = WeatherDetector()
                self._lighting_classifier = LightingClassifier()
                det = self._system.detector
                base_conf = 0.65

            # Model warmup: run a dummy inference to avoid first-frame lag
            try:
                dummy = np.zeros((240, 320, 3), dtype=np.uint8)
                _ = self._system.process_frame(dummy)
            except Exception:
                pass

            print("[PIP] Ready — processing all front camera frames")

            while self.is_running:
                try:
                    frame = self._in_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if frame is None:
                    break

                # Dynamic confidence logic (batched)
                if hasattr(self._system, 'detector'):
                    self._weather_counter += 1
                    if self._weather_counter >= self._weather_frame_skip:
                        small_frame = cv2.resize(frame, (320, 240))
                        w_type, _ = self._weather_detector.detect_weather_conditions(small_frame)
                        l_type, _ = self._lighting_classifier.classify_lighting(small_frame)
                        self._weather_detector.current_weather = w_type
                        self._lighting_classifier.current_lighting = l_type
                        weather_adj = self._weather_detector.get_weather_adjustments()['detection_threshold']
                        lighting_adj = self._lighting_classifier.get_lighting_adjustments()['detection_threshold']
                        dynamic_conf = base_conf * weather_adj * lighting_adj
                        dynamic_conf = np.clip(dynamic_conf, 0.50, 0.80)
                        self._last_weather_type = w_type
                        self._last_lighting_type = l_type
                        self._last_weather_adj = weather_adj
                        self._last_lighting_adj = lighting_adj
                        self._last_dynamic_conf = dynamic_conf
                        self._weather_counter = 0
                    else:
                        dynamic_conf = self._last_dynamic_conf
                        w_type = self._last_weather_type
                        l_type = self._last_lighting_type
                    if hasattr(det, 'conf_threshold'):
                        det.conf_threshold = dynamic_conf
                    if hasattr(det, 'model') and hasattr(det.model, 'conf'):
                        det.model.conf = dynamic_conf
                    self._print_counter += 1
                    if self._print_counter >= self._print_skip:
                        print(f"[PIP] Dynamic conf: {dynamic_conf:.3f} (weather:{w_type}, light:{l_type})")
                        self._print_counter = 0

                if not self.detection_enabled:
                    self._cache_and_push(frame)
                    self.crossing_alerts  = []
                    self.detection_status = False
                    continue

                try:
                    results, _ = self._system.process_frame(frame)

                    # FP Filters + NMS
                    filtered_results = []
                    h_frame, w_frame = frame.shape[:2]
                    min_box_area = (w_frame * h_frame) * 0.0015  # 0.15% (stricter)
                    min_height = h_frame * 0.025  # No tiny distant

                    boxes = []
                    scores = []
                    indices = []
                    for i, res in enumerate(results):
                        bbox = res["bbox"]
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        w_box = x2 - x1
                        h_box = y2 - y1
                        area = w_box * h_box

                        # Filter 1: Area (stricter)
                        if area < min_box_area:
                            continue

                        # Filter 2: Min height
                        if h_box < min_height:
                            continue

                        # Filter 3: Human aspect ratio 0.25-0.7 (w/h)
                        aspect = w_box / max(1, h_box)
                        if not (0.25 <= aspect <= 0.70):
                            continue

                        boxes.append([x1, y1, w_box, h_box])
                        scores.append(res.get("intent_prob", 0.5))  # Use intent as score
                        indices.append(i)

                    # NMS: Suppress overlapping >0.45
                    if len(boxes) > 1:
                        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.45)
                        if len(indices) > 0:
                            indices = indices.flatten()
                        else:
                            indices = []
                    else:
                        indices = list(range(len(boxes)))

                    filtered_results = [results[i] for i in indices]
                    results = filtered_results

                    # Draw ONLY clean bounding boxes — no text, no panels, no banners
                    annotated = frame.copy()
                    high_intent_alerts = []

                    for res in results:
                        bbox       = res["bbox"]
                        intent     = res.get("intent_prob", 0.0)

                        x1, y1, x2, y2 = (int(v) for v in bbox)
                        box_area = (x2 - x1) * (y2 - y1)

                        # Skip boxes that are too small — likely false positives
                        if box_area < min_box_area:
                            continue

                        # Color purely by intent probability
                        if intent >= 0.80:
                            color, thick = (0, 0, 220), 3    # red   — high crossing intent
                        elif intent >= 0.50:
                            color, thick = (0, 140, 255), 2  # orange — moderate intent
                        else:
                            color, thick = (0, 200, 0), 2    # green  — low intent / safe

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

                        # Only flag as alert if intent >= 0.80
                        if intent >= 0.80:
                            high_intent_alerts.append(res)

                    self.crossing_alerts  = high_intent_alerts
                    self.detection_status = len(high_intent_alerts) > 0
                    self.statistics       = self._system.get_statistics()

                except Exception as e:
                    print(f"[PIP] Frame error: {e}")
                    annotated = frame.copy()

                self._cache_and_push(annotated)

        except Exception as e:
            print(f"[PIP] Worker error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False

    def _cache_and_push(self, frame):
        """Cache as last annotated and push to output queue."""
        with self._last_annotated_lock:
            self._last_annotated = frame
        try:
            self._out_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._out_queue.put_nowait(frame)
        except queue.Full:
            pass
