"""
Driver Distraction Detection processor — robust webcam fallback, detection toggle.

Key points
──────────
• _find_predictor() walks up from __file__ to find the canonical
  DriverDistractionDetection/Main/ folder regardless of CWD.
• clean_mode is FALSE so that the HUD, eye-contour overlays, face
  bounding-box and alert pill are all rendered in the driver feed.
• calibration_complete (threading.Event) is set as soon as a profile exists
  OR immediately after the calibration wizard saves one.
• startup_error (str|None) captures any initialisation problem for the UI
  to surface as a custom dark-themed popup.
"""

import cv2
import time
import os
import threading
import sys
from pathlib import Path

from .base_processor import BaseProcessor

# ── Audio alert — single low 800 Hz buzz ─────────────────────────────────────
_beep_lock = threading.Lock()

try:
    import winsound
    def _play_alert():
        winsound.Beep(800, 400)
except ImportError:
    def _play_alert():
        if sys.platform == 'darwin':
            os.system('afplay /System/Library/Sounds/Sosumi.aiff 2>/dev/null')
        else:
            os.system('beep -f 800 -l 400 2>/dev/null || printf "\\a"')

def beep_async():
    if not _beep_lock.acquire(blocking=False):
        return
    def _run():
        try:
            _play_alert()
        finally:
            _beep_lock.release()
    threading.Thread(target=_run, daemon=True).start()


class DriverDistractionProcessor(BaseProcessor):
    """
    Wraps DriverDistractionDetector as a BaseProcessor.

    Public attributes readable after start():
        calibration_complete  – threading.Event, set once calibration is done
                                 (or immediately if profile already exists)
        startup_error         – str or None
        is_calibrated         – bool mirror of the above event
    """

    def __init__(self, video_source=0,
                 predictor_path='',
                 profile_path=''):
        super().__init__(video_source)
        self.predictor_path = predictor_path
        self.profile_path   = profile_path

        # Detection results exposed to the UI
        self.alerts       = []
        self.eyes_open    = True
        self.gaze         = 'straight'
        self.yaw_status   = 'straight'
        self.avg_ear      = 0.28
        self.distracted   = False

        self.detection_enabled = True

        # Calibration signalling
        self.calibration_complete = threading.Event()
        self.is_calibrated        = False
        self.startup_error        = None

    # ──────────────────────────────────────────────────────────────────────────
    # Path resolution
    # ──────────────────────────────────────────────────────────────────────────

    def _find_predictor(self):
        """
        Locate shape_predictor_68_face_landmarks.dat.

        Expected layout:
          NeuroDrive/
            DriverDistractionDetection/Main/
              shape_predictor_68_face_landmarks.dat   <- canonical
            Main/processors/
              driver_distraction_processor.py         <- this file
        """
        this_file  = Path(__file__).resolve()
        main_dir   = this_file.parent.parent          # .../NeuroDrive/Main/
        neuro_root = main_dir.parent                  # .../NeuroDrive/
        canonical  = (neuro_root / "DriverDistractionDetection"
                      / "Main" / "shape_predictor_68_face_landmarks.dat")

        candidates = [
            self.predictor_path,
            str(canonical),
            str(this_file.parent / 'shape_predictor_68_face_landmarks.dat'),
            str(main_dir / 'shape_predictor_68_face_landmarks.dat'),
            str(neuro_root / 'shape_predictor_68_face_landmarks.dat'),
            '../shape_predictor_68_face_landmarks.dat',
            'shape_predictor_68_face_landmarks.dat',
        ]
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    def _resolve_profile_path(self):
        """Return an absolute path for the driver profile JSON."""
        if self.profile_path and os.path.isabs(self.profile_path):
            return self.profile_path
        this_file  = Path(__file__).resolve()
        neuro_root = this_file.parent.parent.parent
        return str(neuro_root / "DriverDistractionDetection" / "Main" / "driver_profile.json")

    # ──────────────────────────────────────────────────────────────────────────
    # Camera open
    # ──────────────────────────────────────────────────────────────────────────

    def _open_cap(self):
        source = self.video_source
        if isinstance(source, str) and source not in ('0', ''):
            if not os.path.exists(source):
                print(f"[DriverCam] File not found: {source} — trying webcam 0")
                source = 0
        elif isinstance(source, str) and source == '0':
            source = 0

        cap = cv2.VideoCapture(source)
        if not cap.isOpened() and source != 0:
            print(f"[DriverCam] Can't open {source} — falling back to webcam 0")
            cap.release()
            cap = cv2.VideoCapture(0)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS,          30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        return cap

    # ──────────────────────────────────────────────────────────────────────────
    # Main processing thread
    # ──────────────────────────────────────────────────────────────────────────

    def _run(self):
        # Ensure profile path is absolute
        resolved_profile = self._resolve_profile_path()
        if not self.profile_path or not os.path.isabs(self.profile_path):
            self.profile_path = resolved_profile
        print(f"[DriverCam] Profile path : {self.profile_path}")

        cap = self._open_cap()
        if not cap.isOpened():
            msg = "No camera available — driver feed disabled."
            print(f"[DriverCam] {msg}")
            self.startup_error = msg
            self.calibration_complete.set()
            self.is_running = False
            return

        print("[DriverCam] Camera opened — starting feed.")

        # ── Locate shape predictor ────────────────────────────────────────────
        dat = self._find_predictor()

        # ── Load detector ─────────────────────────────────────────────────────
        detector        = None
        use_full_detect = False

        if dat:
            print(f"[DriverCam] Found predictor at: {dat}")
            try:
                ddd_main = str(Path(dat).parent)
                if ddd_main not in sys.path:
                    sys.path.insert(0, ddd_main)

                from main import DriverDistractionDetector
                detector = DriverDistractionDetector(
                    predictor_path=dat,
                    profile_path=self.profile_path,
                )
                # ── clean_mode = False so ALL overlays render ──────────────────
                # (HUD panel, eye contours, landmark dots, alert pill, face box)
                detector.clean_mode = True
                use_full_detect = True
                print("[DriverCam] Full distraction detector loaded.")
            except Exception as e:
                msg = f"Detector load failed: {e}"
                print(f"[DriverCam] {msg}")
                self.startup_error = msg
        else:
            msg = "shape_predictor_68_face_landmarks.dat not found — video-only mode."
            print(f"[DriverCam] {msg}")
            self.startup_error = msg

        # Signal calibration immediately if already calibrated
        if use_full_detect and detector is not None and detector.calibrated:
            self.is_calibrated = True
            self.calibration_complete.set()
            print("[DriverCam] Profile already exists — calibration not required.")
        elif not use_full_detect:
            self.calibration_complete.set()

        # ── Per-frame state ───────────────────────────────────────────────────
        last_face     = None
        frame_count   = 0
        DETECT_EVERY  = 4
        _last_beep    = 0.0
        BEEP_COOLDOWN = 4.0

        print("[DriverCam] Started.")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                source = self.video_source
                if isinstance(source, str) and source not in ('0', '') and os.path.exists(source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # ── Detection disabled ────────────────────────────────────────────
            if not self.detection_enabled:
                self.distracted = False
                self.alerts     = []
                self._put_frame(frame)
                continue

            # ── Full distraction detection ────────────────────────────────────
            if use_full_detect and detector is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if frame_count % DETECT_EVERY == 0 or last_face is None:
                        import dlib
                        small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
                        faces = detector.detector(small, 0)
                        if faces:
                            f = max(faces, key=lambda r: r.width() * r.height())
                            last_face = dlib.rectangle(
                                f.left()*2, f.top()*2, f.right()*2, f.bottom()*2)
                        else:
                            last_face = None

                    if last_face is None:
                        detector._put(frame, "No face detected", (20, 60),
                                      detector.C_ALERT, 1.0, 2)
                        self.alerts     = []
                        self.distracted = False
                        self._put_frame(frame)
                        continue

                    shape     = detector.predictor(gray, last_face)
                    landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

                    # ── Calibration wizard ────────────────────────────────────
                    if not detector.calibrated:
                        if not detector._intro_shown:
                            detector.draw_intro_screen(frame)
                            self._put_frame(frame)
                            if not hasattr(self, '_intro_start'):
                                self._intro_start = time.time()
                            if time.time() - self._intro_start >= 3.0:
                                detector._intro_shown = True
                            time.sleep(0.03)
                            continue

                        from main import PHASES, COLLECT_FRAMES, SHOW_DONE_MS
                        if detector._phase_idx < len(PHASES):
                            if detector._done_shown_at is None:
                                detector._calib_collect(gray, landmarks, frame.shape)
                                if len(detector._phase_samples) >= COLLECT_FRAMES:
                                    detector._calib_finish_phase()
                            else:
                                if (time.time() - detector._done_shown_at) * 1000 >= SHOW_DONE_MS:
                                    detector._calib_advance_phase()

                        detector.draw_calibration_screen(frame, detector._done_shown_at is not None)
                        self._put_frame(frame)

                        if detector.calibrated and not self.is_calibrated:
                            self.is_calibrated = True
                            self.calibration_complete.set()
                            print("[DriverCam] Calibration complete — signalling main thread.")

                        time.sleep(0.03)
                        continue

                    # ── Normal detection ──────────────────────────────────────
                    results = detector.analyze(frame, gray, landmarks)

                    self.alerts           = results['alerts']
                    self.distracted       = len(self.alerts) > 0
                    self.eyes_open        = results['left_eye_open'] and results['right_eye_open']
                    self.gaze             = results['gaze']
                    self.yaw_status       = results['yaw_status']
                    self.avg_ear          = results['avg_ear']
                    self.detection_status = self.distracted

                    if self.distracted:
                        now = time.time()
                        if now - _last_beep >= BEEP_COOLDOWN:
                            _last_beep = now
                            beep_async()

                    # ── Draw only the alert pill when distracted ──────────────
                    # (no HUD, no face box, no landmark overlays)
                    if results['alerts']:
                        detector.draw_alert_pill(frame, results['alerts'])

                except Exception as e:
                    print(f"[DriverCam] Detection error: {e}")

            else:
                # ── Video-only / no-predictor mode ────────────────────────────
                h, w = frame.shape[:2]
                cv2.putText(frame, "Driver Camera", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 210, 70), 2, cv2.LINE_AA)
                if not dat:
                    cv2.putText(frame, "shape_predictor not found", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 200), 2, cv2.LINE_AA)

            self._put_frame(frame)

        cap.release()
        print("[DriverCam] Stopped.")