"""
Keys
  Q   quit
  S   save snapshot
  D   toggle clean / detail mode
  R   force re-calibration (deletes saved profile)
  Mouse-click top-right button -> same as D
"""

import cv2
import numpy as np
import dlib
from scipy.spatial import distance
import time
import threading
import json
import os
from datetime import datetime

# ── Audio alert ───────────────────────────────────────────────────────────────
# Two sharp high-pitched bursts — urgent double-beep, fires once per event.
# _beep_lock prevents overlap if somehow called concurrently.
_beep_lock = threading.Lock()

try:
    import winsound
    def _play_alert():
        # Two sharp descending tones — sounds urgent, not annoying
        winsound.Beep(1800, 120)
        time.sleep(0.04)
        winsound.Beep(1400, 200)
except ImportError:
    import sys
    def _play_alert():
        if sys.platform == 'darwin':
            # macOS: two quick tones via afplay pitch shift
            os.system('afplay /System/Library/Sounds/Glass.aiff -rate 1.4 2>/dev/null')
            time.sleep(0.05)
            os.system('afplay /System/Library/Sounds/Glass.aiff -rate 1.1 2>/dev/null')
        else:
            # Linux: two beeps at different frequencies
            os.system(
                'beep -f 1800 -l 120 -D 40 -f 1400 -l 200 2>/dev/null || '
                '(printf "\\a"; sleep 0.15; printf "\\a")'
            )

def beep_async():
    """Fire the alert sound once. Silently ignored if already playing."""
    if not _beep_lock.acquire(blocking=False):
        return
    def _run():
        try:
            _play_alert()
        finally:
            _beep_lock.release()
    threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION WIZARD PHASES
# Each tuple: (phase_id, headline, sub-instruction, BGR colour hint)
# ─────────────────────────────────────────────────────────────────────────────
PHASES = [
    ('neutral',          'Look STRAIGHT AHEAD',    'Eyes open, face forward',          (0, 210,  70)),
    ('gaze_left',        'Look LEFT',              'Eyes only - keep head still',      (0, 180, 255)),
    ('gaze_right',       'Look RIGHT',             'Eyes only - keep head still',      (0, 180, 255)),
    ('eye_left_close',   'Close LEFT eye only',    'Keep your right eye open',         (60, 130, 255)),
    ('eye_right_close',  'Close RIGHT eye only',   'Keep your left eye open',          (60, 130, 255)),
    ('head_left',        'Turn HEAD LEFT',         'Rotate head - not just eyes',      (0, 180, 255)),
    ('head_right',       'Turn HEAD RIGHT',        'Rotate head - not just eyes',      (0, 180, 255)),
]

COLLECT_FRAMES = 120  # frames of data collected per phase  (~4 s at 30 fps)
SHOW_DONE_MS   = 900  # milliseconds to show "Captured!" before advancing


# ─────────────────────────────────────────────────────────────────────────────
class DriverDistractionDetector:

    # BGR colours
    C_OK     = (0,   210,  70)
    C_WARN   = (0,   165, 255)
    C_ALERT  = (30,   30, 220)
    C_DIM    = (120, 120, 120)
    C_ACCENT = (50,  200, 255)
    C_WHITE  = (230, 230, 230)
    C_BG     = (18,   18,  18)

    def __init__(self, predictor_path='shape_predictor_68_face_landmarks.dat',
                 profile_path='driver_profile.json'):

        self.detector     = dlib.get_frontal_face_detector()
        self.predictor    = dlib.shape_predictor(predictor_path)
        self.profile_path = profile_path

        self.LEFT_EYE  = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))

        # ── Detection timing ──────────────────────────────────────────────────
        self.ALERT_HOLD_SECS = 2.0

        # Beep fires ONCE when a condition first crosses ALERT_HOLD_SECS.
        # It cannot fire again for that condition until:
        #   (a) the condition has fully cleared, AND
        #   (b) BEEP_RESET_SECS have passed since it cleared.
        self.BEEP_RESET_SECS = 5.0

        # ── Calibrated thresholds (defaults; overwritten by wizard or profile) ─
        self.EAR_L_THRESH      = 0.20
        self.EAR_R_THRESH      = 0.20
        self.GAZE_LO           = 0.65
        self.GAZE_HI           = 1.50
        self.GAZE_NEUTRAL      = 1.0
        self.YAW_L_THRESH      = -18.0
        self.YAW_R_THRESH      =  18.0
        self.YAW_NEUTRAL       = 0.0

        # ── Calibration wizard state ───────────────────────────────────────────
        self.calibrated       = False
        self._phase_idx       = 0
        self._phase_samples   = []
        self._phase_data      = {}
        self._done_shown_at   = None
        self._intro_shown     = False   # True once user presses SPACE to begin

        # ── Per-condition sustain timers ───────────────────────────────────────
        self._cond_start = {k: None for k in
                            ['eyes_closed', 'gaze_left', 'gaze_right',
                             'head_left',   'head_right']}

        self._beep_state     = {k: 'ready' for k in self._cond_start}
        self._beep_clear_at  = {k: None    for k in self._cond_start}

        # ── Smoothing (exponential moving average on raw sensor values) ──────
        # Alpha closer to 0 = more smoothing, closer to 1 = more responsive.
        self.EMA_ALPHA   = 0.15   # strong smoothing — eliminates jitter
        self._sm_le      = None   # smoothed left EAR
        self._sm_re      = None   # smoothed right EAR
        self._sm_gaze    = None   # smoothed gaze ratio
        self._sm_yaw     = None   # smoothed yaw angle

        # Display label vote buffer: majority over N frames before label flips
        self._VOTE_N     = 12
        self._vote_gaze  = []
        self._vote_yaw   = []
        self._vote_eyes  = []
        self._disp_gaze  = 'straight'
        self._disp_yaw   = 'straight'
        self._disp_eyes  = True   # True = open

        # ── Head-pose 3-D model ───────────────────────────────────────────────
        self.model_points = np.array([
            (  0.0,    0.0,    0.0),
            (  0.0, -330.0,  -65.0),
            (-225.0,  170.0, -135.0),
            ( 225.0,  170.0, -135.0),
            (-150.0, -150.0, -125.0),
            ( 150.0, -150.0, -125.0),
        ], dtype=np.float64)
        self.camera_matrix = None

        self.clean_mode = False
        self._btn_rect  = None

        self._load_profile()

    # ─────────────────────────────────────────────────────────────────────────
    # PROFILE SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────────────
    def _profile_dict(self):
        return {
            'EAR_L_THRESH': self.EAR_L_THRESH,
            'EAR_R_THRESH': self.EAR_R_THRESH,
            'GAZE_LO':      self.GAZE_LO,
            'GAZE_HI':      self.GAZE_HI,
            'GAZE_NEUTRAL': self.GAZE_NEUTRAL,
            'YAW_L_THRESH': self.YAW_L_THRESH,
            'YAW_R_THRESH': self.YAW_R_THRESH,
            'YAW_NEUTRAL':  self.YAW_NEUTRAL,
            'calibrated_at': datetime.now().isoformat(timespec='seconds'),
        }

    def _save_profile(self):
        try:
            with open(self.profile_path, 'w') as f:
                json.dump(self._profile_dict(), f, indent=2)
            print(f"[Profile] Saved -> {self.profile_path}")
        except Exception as e:
            print(f"[Profile] Save error: {e}")

    def _load_profile(self):
        if not os.path.exists(self.profile_path):
            print("[Profile] No saved profile — calibration wizard will run.")
            return
        try:
            with open(self.profile_path) as f:
                d = json.load(f)
            for k in self._profile_dict():
                if k != 'calibrated_at' and k in d:
                    setattr(self, k, float(d[k]))
            self.calibrated = True
            print(f"[Profile] Loaded (calibrated {d.get('calibrated_at','?')})")
            print(f"  EAR L={self.EAR_L_THRESH:.3f} R={self.EAR_R_THRESH:.3f}  "
                  f"Gaze [{self.GAZE_LO:.2f} - {self.GAZE_HI:.2f}]  "
                  f"Yaw L={self.YAW_L_THRESH:.1f} R={self.YAW_R_THRESH:.1f}")
        except Exception as e:
            print(f"[Profile] Load error ({e}) — will recalibrate.")

    def reset_calibration(self):
        if os.path.exists(self.profile_path):
            os.remove(self.profile_path)
        self.calibrated     = False
        self._phase_idx     = 0
        self._phase_samples = []
        self._phase_data    = {}
        self._done_shown_at = None
        self._intro_shown   = False
        self._sm_le = self._sm_re = self._sm_gaze = self._sm_yaw = None
        self._vote_gaze = []; self._vote_yaw = []; self._vote_eyes = []
        print("[Profile] Reset — calibration wizard starting.")

    # ─────────────────────────────────────────────────────────────────────────
    # RAW SENSOR READINGS
    # ─────────────────────────────────────────────────────────────────────────
    def _ema(self, prev, new):
        """Exponential moving average. Seeds from first value."""
        if prev is None:
            return new
        return self.EMA_ALPHA * new + (1.0 - self.EMA_ALPHA) * prev

    def _vote(self, buf, new_val):
        """Append to vote buffer, return majority value."""
        buf.append(new_val)
        if len(buf) > self._VOTE_N:
            buf.pop(0)
        return max(set(buf), key=buf.count)

    def _ear(self, pts):
        A = distance.euclidean(pts[1], pts[5])
        B = distance.euclidean(pts[2], pts[4])
        C = distance.euclidean(pts[0], pts[3])
        return (A + B) / (2.0 * C) if C else 0.0

    def _raw_ears(self, landmarks):
        le = self._ear([landmarks[i] for i in self.LEFT_EYE])
        re = self._ear([landmarks[i] for i in self.RIGHT_EYE])
        return le, re

    def _gaze_one_eye(self, gray, indices, landmarks):
        region = np.array([landmarks[i] for i in indices], dtype=np.int32)
        mask   = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [region], 255)
        roi    = cv2.bitwise_and(gray, gray, mask=mask)
        x, y, w, h = cv2.boundingRect(region)
        if w == 0 or h == 0:
            return 1.0
        crop  = roi[y:y+h, x:x+w]
        mc    = mask[y:y+h, x:x+w]
        _, th = cv2.threshold(crop, 70, 255, cv2.THRESH_BINARY_INV)
        th    = cv2.bitwise_and(th, th, mask=mc)
        lw    = cv2.countNonZero(th[:, :w // 2])
        rw    = cv2.countNonZero(th[:, w // 2:])
        return lw / rw if rw else 6.0

    def _raw_gaze(self, gray, landmarks):
        lr = self._gaze_one_eye(gray, self.LEFT_EYE,  landmarks)
        rr = self._gaze_one_eye(gray, self.RIGHT_EYE, landmarks)
        return (lr + rr) / 2.0, lr, rr

    def _build_cam(self, w, h):
        return np.array([[w, 0, w/2.], [0, w, h/2.], [0, 0, 1.]], dtype=np.float64)

    def _raw_yaw(self, landmarks, fshape):
        fh, fw = fshape[:2]
        if self.camera_matrix is None or self.camera_matrix[0, 2] != fw / 2:
            self.camera_matrix = self._build_cam(fw, fh)
        pts = np.array([landmarks[i] for i in [30, 8, 36, 45, 48, 54]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            self.model_points, pts, self.camera_matrix, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(np.hstack([rmat, tvec]))
        return float(np.squeeze(euler)[1])

    # ─────────────────────────────────────────────────────────────────────────
    # CALIBRATION WIZARD
    # ─────────────────────────────────────────────────────────────────────────
    def _calib_collect(self, gray, landmarks, fshape):
        le, re     = self._raw_ears(landmarks)
        gavg, _, _ = self._raw_gaze(gray, landmarks)
        yaw        = self._raw_yaw(landmarks, fshape)
        self._phase_samples.append({'le': le, 're': re, 'gaze': gavg, 'yaw': yaw})

    def _calib_finish_phase(self):
        pid = PHASES[self._phase_idx][0]
        s   = self._phase_samples
        self._phase_data[pid] = {
            'le':   float(np.mean([x['le']   for x in s])),
            're':   float(np.mean([x['re']   for x in s])),
            'gaze': float(np.mean([x['gaze'] for x in s])),
            'yaw':  float(np.mean([x['yaw']  for x in s])),
        }
        print(f"[Calib] {pid}: le={self._phase_data[pid]['le']:.3f}  "
              f"re={self._phase_data[pid]['re']:.3f}  "
              f"gaze={self._phase_data[pid]['gaze']:.3f}  "
              f"yaw={self._phase_data[pid]['yaw']:.1f}")
        self._done_shown_at = time.time()

    def _calib_advance_phase(self):
        self._phase_samples = []
        self._done_shown_at = None
        self._phase_idx    += 1
        if self._phase_idx >= len(PHASES):
            self._compute_thresholds()

    def _compute_thresholds(self):
        d = self._phase_data

        ear_open_l = d['neutral']['le']
        ear_open_r = d['neutral']['re']
        ear_cls_l  = d['eye_left_close']['le']
        ear_cls_r  = d['eye_right_close']['re']
        self.EAR_L_THRESH = (ear_open_l + ear_cls_l) / 2.0 * 0.92
        self.EAR_R_THRESH = (ear_open_r + ear_cls_r) / 2.0 * 0.92

        g_neutral = d['neutral']['gaze']
        g_left    = d['gaze_left']['gaze']
        g_right   = d['gaze_right']['gaze']
        self.GAZE_NEUTRAL = g_neutral
        # Frame is mirrored: looking left makes the dark pupil appear on the right
        # side of the eye region, giving a LOW ratio. So LO threshold = right, HI = left.
        # We swap the phase data to match natural direction labelling.
        self.GAZE_LO = g_neutral - 0.55 * abs(g_neutral - g_right)   # low ratio -> looking right
        self.GAZE_HI = g_neutral + 0.55 * abs(g_left - g_neutral)    # high ratio -> looking left

        y_neutral = d['neutral']['yaw']
        y_left    = d['head_left']['yaw']
        y_right   = d['head_right']['yaw']
        self.YAW_NEUTRAL  = y_neutral
        self.YAW_L_THRESH = y_neutral - 0.55 * abs(y_neutral - y_left)
        self.YAW_R_THRESH = y_neutral + 0.55 * abs(y_right - y_neutral)

        self.calibrated = True
        self._save_profile()
        print("[Calib] Done!")
        print(f"  EAR_L={self.EAR_L_THRESH:.3f}  EAR_R={self.EAR_R_THRESH:.3f}")
        print(f"  Gaze neutral={g_neutral:.2f} lo={self.GAZE_LO:.2f} hi={self.GAZE_HI:.2f}")
        print(f"  Yaw neutral={y_neutral:.1f} L_thr={self.YAW_L_THRESH:.1f} R_thr={self.YAW_R_THRESH:.1f}")

    # ─────────────────────────────────────────────────────────────────────────
    # DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    def eyes_open_closed(self, landmarks):
        le_raw, re_raw = self._raw_ears(landmarks)
        self._sm_le = self._ema(self._sm_le, le_raw)
        self._sm_re = self._ema(self._sm_re, re_raw)
        le, re = self._sm_le, self._sm_re
        raw_open = (le > self.EAR_L_THRESH) and (re > self.EAR_R_THRESH)
        self._disp_eyes = self._vote(self._vote_eyes, raw_open)
        return self._disp_eyes, self._disp_eyes, (le+re)/2.0, le, re

    def gaze_direction(self, gray, landmarks):
        avg_raw, lr, rr = self._raw_gaze(gray, landmarks)
        self._sm_gaze = self._ema(self._sm_gaze, avg_raw)
        avg = self._sm_gaze
        if   avg < self.GAZE_LO: raw_dir = 'right'
        elif avg > self.GAZE_HI: raw_dir = 'left'
        else:                     raw_dir = 'straight'
        self._disp_gaze = self._vote(self._vote_gaze, raw_dir)
        return self._disp_gaze, self._sm_gaze, self._sm_gaze

    def head_yaw(self, landmarks, fshape):
        yaw_raw = self._raw_yaw(landmarks, fshape)
        self._sm_yaw = self._ema(self._sm_yaw, yaw_raw)
        yaw = self._sm_yaw
        if   yaw < self.YAW_L_THRESH: raw_status = 'looking_left'
        elif yaw > self.YAW_R_THRESH: raw_status = 'looking_right'
        else:                          raw_status = 'straight'
        self._disp_yaw = self._vote(self._vote_yaw, raw_status)
        return yaw - self.YAW_NEUTRAL, self._disp_yaw

    # ─────────────────────────────────────────────────────────────────────────
    # TIMER + BEEP STATE MACHINE
    # ─────────────────────────────────────────────────────────────────────────
    def _update(self, key, active, now):
        state = self._beep_state[key]

        if state == 'cooling':
            clear_at = self._beep_clear_at[key]
            if clear_at is not None and (now - clear_at) >= self.BEEP_RESET_SECS:
                self._beep_state[key]    = 'ready'
                self._beep_clear_at[key] = None
                state = 'ready'

        if active:
            if self._cond_start[key] is None:
                self._cond_start[key] = now

            elapsed = now - self._cond_start[key]
            fired   = elapsed >= self.ALERT_HOLD_SECS

            if fired and state == 'ready':
                beep_async()
                self._beep_state[key] = 'fired'

            return fired

        else:
            self._cond_start[key] = None

            if state == 'fired':
                self._beep_state[key]    = 'cooling'
                self._beep_clear_at[key] = now

            return False

    def _remaining(self, key, now):
        if self._cond_start[key] is None:
            return self.ALERT_HOLD_SECS
        return max(0.0, self.ALERT_HOLD_SECS - (now - self._cond_start[key]))

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    def analyze(self, frame, gray, landmarks):
        now = time.time()

        eye_span = abs(landmarks[45][0] - landmarks[36][0])
        jaw_span = abs(landmarks[16][0] - landmarks[0][0])
        if jaw_span > 0 and (eye_span / jaw_span) < 0.22:
            safe_t = {k: self.ALERT_HOLD_SECS for k in self._cond_start}
            return dict(
                left_eye_open=True, right_eye_open=True,
                avg_ear=0.28, le_ear=0.28, re_ear=0.28,
                gaze='straight', gaze_l=1.0, gaze_r=1.0,
                yaw=0.0, yaw_status='straight',
                alerts=['FACE OBSCURED'], timers=safe_t, now=now, occluded=True,
            )

        lo, ro, avg_ear, le_ear, re_ear = self.eyes_open_closed(landmarks)
        gaze, gl, gr    = self.gaze_direction(gray, landmarks)
        yaw, yaw_status = self.head_yaw(landmarks, frame.shape)

        checks = {
            'eyes_closed': (not lo) or (not ro),
            'gaze_left':   gaze == 'left',
            'gaze_right':  gaze == 'right',
            'head_left':   yaw_status == 'looking_left',
            'head_right':  yaw_status == 'looking_right',
        }
        labels = {
            'eyes_closed': 'EYES CLOSED',
            'gaze_left':   'GAZE LEFT',
            'gaze_right':  'GAZE RIGHT',
            'head_left':   'HEAD TURNED LEFT',
            'head_right':  'HEAD TURNED RIGHT',
        }
        alerts, timers = [], {}
        for key, active in checks.items():
            fired       = self._update(key, active, now)
            timers[key] = self._remaining(key, now)
            if fired:
                alerts.append(labels[key])

        return dict(
            left_eye_open=lo, right_eye_open=ro,
            avg_ear=avg_ear, le_ear=le_ear, re_ear=re_ear,
            gaze=gaze, gaze_l=gl, gaze_r=gr,
            yaw=yaw, yaw_status=yaw_status,
            alerts=alerts, timers=timers, now=now, occluded=False,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # DRAWING HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _put(self, frame, txt, pos, color=None, scale=0.58, thick=1):
        # OpenCV's built-in fonts only support ASCII. Strip anything above 127
        # so non-ASCII characters never render as ??? boxes.
        safe = txt.encode('ascii', errors='replace').decode('ascii').replace('?', '')
        cv2.putText(frame, safe, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color or self.C_WHITE, thick, cv2.LINE_AA)

    def _bar(self, frame, x, y, w, h, frac, fg):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (45, 45, 45), -1)
        fill = int(w * min(1.0, max(0.0, frac)))
        if fill:
            cv2.rectangle(frame, (x, y), (x+fill, y+h), fg, -1)

    def draw_toggle_button(self, frame):
        fh, fw = frame.shape[:2]
        bw, bh, mg = 130, 36, 14
        x1, y1 = fw - bw - mg, mg
        x2, y2 = fw - mg, y1 + bh
        self._btn_rect = (x1, y1, x2, y2)
        ovr = frame.copy()
        cv2.rectangle(ovr, (x1, y1), (x2, y2),
                      (30, 80, 30) if self.clean_mode else (35, 35, 35), -1)
        cv2.addWeighted(ovr, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
        label  = "SHOW HUD"  if self.clean_mode else "CLEAN VIEW"
        lcolor = self.C_OK   if self.clean_mode else (160, 160, 160)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        self._put(frame, label, (x1 + (bw-tw)//2, y1+24), lcolor, 0.48)

    def draw_alert_pill(self, frame, alerts):
        if not alerts:
            return
        fh, fw = frame.shape[:2]
        text = alerts[0] if len(alerts) == 1 else f"{alerts[0]} (+{len(alerts)-1})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)
        pad = 20
        pw, ph = tw + pad*2, th + pad + 8
        px, py = (fw - pw) // 2, fh - ph - 36
        alpha = 0.55 + 0.35 * abs(np.sin(time.time() * 3.5))
        ovr = frame.copy()
        cv2.rectangle(ovr, (px, py), (px+pw, py+ph), (20, 20, 200), -1)
        cv2.addWeighted(ovr, alpha, frame, 1-alpha, 0, frame)
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), (80, 80, 255), 2)
        self._put(frame, text, (px+pad, py+ph-pad//2), (255, 255, 255), 0.80, 2)

    def draw_overlays(self, frame, landmarks):
        for idx in [self.LEFT_EYE, self.RIGHT_EYE]:
            pts = np.array([landmarks[i] for i in idx], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 200, 255), 1)
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 220, 80), -1)

    # ─────────────────────────────────────────────────────────────────────────
    # CALIBRATION INTRO SCREEN
    # ─────────────────────────────────────────────────────────────────────────
    def draw_intro_screen(self, frame):
        fh, fw = frame.shape[:2]
        ovr = frame.copy()
        cv2.rectangle(ovr, (0, 0), (fw, fh), (0, 0, 0), -1)
        cv2.addWeighted(ovr, 0.72, frame, 0.28, 0, frame)

        put = self._put
        cx  = fw // 2

        # Title
        title = "CALIBRATION SETUP"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.10, 2)
        put(frame, title, (cx - tw//2, 80), self.C_ACCENT, 1.10, 2)

        # Subtitle
        sub = "Before we start, here's what to expect:"
        (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 1)
        put(frame, sub, (cx - tw//2, 122), (180, 180, 180), 0.60)

        cv2.line(frame, (fw//6, 136), (fw*5//6, 136), (55, 55, 55), 1)

        # Steps list
        steps = [
            ("1", "LOOK STRAIGHT",    "Face forward, eyes open - sets your neutral baseline"),
            ("2", "LOOK LEFT / RIGHT","Eyes only, keep head still - calibrates gaze detection"),
            ("3", "CLOSE EACH EYE",   "Close one eye at a time - calibrates eye-closed threshold"),
            ("4", "TURN HEAD L / R",  "Rotate your full head left then right - calibrates head tracking"),
        ]

        y = 168
        for num, heading, detail in steps:
            cv2.circle(frame, (fw//6 + 18, y - 6), 16, self.C_ACCENT, -1)
            put(frame, num,     (fw//6 + 13, y - 1), (0, 0, 0),       0.52, 2)
            put(frame, heading, (fw//6 + 44, y - 2), self.C_WHITE,     0.62)
            put(frame, detail,  (fw//6 + 44, y + 20),(160, 160, 160),  0.46)
            y += 64

        cv2.line(frame, (fw//6, y), (fw*5//6, y), (55, 55, 55), 1)
        y += 20

        # Tips
        tips = [
            "  Each step takes ~4 seconds - hold the pose until the bar fills",
            "  Sit in your normal driving position before starting",
            "  Good lighting on your face = better accuracy",
            "  Profile is saved - you only calibrate once per driver",
        ]
        for tip in tips:
            put(frame, tip, (fw//6, y), (140, 200, 140), 0.46)
            y += 24

        # SPACE prompt (pulsing)
        alpha_pulse = 0.6 + 0.4 * abs(np.sin(time.time() * 2.2))
        prompt = "Press  SPACE  to begin"
        (tw, th), _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)
        px, py = cx - tw//2, fh - 50
        pulse_col = tuple(int(c * alpha_pulse) for c in (80, 220, 80))
        put(frame, prompt, (px, py), pulse_col, 0.80, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # CALIBRATION WIZARD SCREEN
    # ─────────────────────────────────────────────────────────────────────────
    def draw_calibration_screen(self, frame, showing_done):
        fh, fw   = frame.shape[:2]
        n_phases = len(PHASES)
        pidx     = self._phase_idx
        if pidx >= n_phases:
            return

        phase_id, headline, subline, hint_col = PHASES[pidx]
        n_collected = len(self._phase_samples)

        ovr = frame.copy()
        cv2.rectangle(ovr, (0, 0), (fw, fh), (0, 0, 0), -1)
        cv2.addWeighted(ovr, 0.50, frame, 0.50, 0, frame)

        put = self._put

        step_txt = f"CALIBRATION  Step {pidx+1} of {n_phases}"
        (tw, _), _ = cv2.getTextSize(step_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 1)
        put(frame, step_txt, ((fw-tw)//2, 42), self.C_DIM, 0.60)

        dot_r, gap = 9, 28
        total_w = n_phases * (dot_r*2) + (n_phases-1) * (gap - dot_r*2)
        dx = (fw - total_w) // 2
        for i in range(n_phases):
            if   i < pidx:  col = self.C_OK
            elif i == pidx: col = hint_col
            else:            col = (55, 55, 55)
            cv2.circle(frame, (dx + i * gap, 72), dot_r, col, -1)

        cy = fh // 2 - 65
        (tw, th), _ = cv2.getTextSize(headline, cv2.FONT_HERSHEY_SIMPLEX, 1.10, 2)
        put(frame, headline, ((fw-tw)//2, cy), hint_col, 1.10, 2)
        cy += th + 16

        (tw, th), _ = cv2.getTextSize(subline, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        put(frame, subline, ((fw-tw)//2, cy), (180, 180, 180), 0.65)
        cy += th + 34

        if showing_done:
            done_txt = "Captured!"
            (tw, _), _ = cv2.getTextSize(done_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            put(frame, done_txt, ((fw-tw)//2, cy+10), self.C_OK, 1.0, 2)
        else:
            frac  = min(1.0, n_collected / COLLECT_FRAMES)
            bar_x = fw // 5
            bar_w = fw * 3 // 5
            cv2.rectangle(frame, (bar_x, cy), (bar_x+bar_w, cy+20), (50, 50, 50), -1)
            fill = int(bar_w * frac)
            if fill:
                cv2.rectangle(frame, (bar_x, cy), (bar_x+fill, cy+20), hint_col, -1)
            pct_txt = f"Hold still...  {int(frac*100)}%"
            (tw, _), _ = cv2.getTextSize(pct_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
            put(frame, pct_txt, ((fw-tw)//2, cy+44), self.C_DIM, 0.58)

        put(frame, "R = restart calibration", (14, fh-16), self.C_DIM, 0.44)

    # ─────────────────────────────────────────────────────────────────────────
    # FULL HUD
    # ─────────────────────────────────────────────────────────────────────────
    def draw_hud(self, frame, results):
        fh, fw = frame.shape[:2]
        # HUD panel width scales with frame width
        panel_w = max(300, fw // 4)
        ovr = frame.copy()
        cv2.rectangle(ovr, (0, 0), (panel_w, fh), self.C_BG, -1)
        cv2.addWeighted(ovr, 0.55, frame, 0.45, 0, frame)
        cv2.line(frame, (panel_w, 0), (panel_w, fh), (50, 50, 50), 1)

        put = self._put
        put(frame, "DRIVER MONITOR", (14, 34), self.C_ACCENT, 0.80, 2)
        cv2.line(frame, (14, 42), (panel_w - 12, 42), (55, 55, 55), 1)
        y = 66

        # ── Eyes ──────────────────────────────────────────────────────────────
        put(frame, "EYES", (14, y), self.C_DIM, 0.52);  y += 24
        for side, opn, ear_v, thr in [
            ("LEFT ", results['left_eye_open'],  results['le_ear'], self.EAR_L_THRESH),
            ("RIGHT", results['right_eye_open'], results['re_ear'], self.EAR_R_THRESH),
        ]:
            lbl, col = ("OPEN",   self.C_OK) if opn else ("CLOSED", self.C_ALERT)
            cv2.circle(frame, (24, y-6), 6, col, -1)
            put(frame, f"{side}  {lbl}  ({ear_v:.3f})", (38, y), col, 0.58)
            y += 26
        put(frame, f"thr L={self.EAR_L_THRESH:.3f}  R={self.EAR_R_THRESH:.3f}",
            (16, y), self.C_DIM, 0.46);  y += 12
        if self._cond_start['eyes_closed'] is not None:
            frac = 1.0 - results['timers']['eyes_closed'] / self.ALERT_HOLD_SECS
            self._bar(frame, 16, y, 220, 8, frac, self.C_ALERT)
            elapsed = self.ALERT_HOLD_SECS - results['timers']['eyes_closed']
            put(frame, f"{elapsed:.1f}s", (244, y+8), self.C_DIM, 0.42)
        y += 22
        cv2.line(frame, (14, y), (panel_w-12, y), (40, 40, 40), 1);  y += 14

        # ── Gaze ──────────────────────────────────────────────────────────────
        put(frame, "GAZE", (14, y), self.C_DIM, 0.52);  y += 24
        gaze = results['gaze']
        put(frame, gaze.upper(), (16, y),
            self.C_OK if gaze == 'straight' else self.C_WARN, 0.70);  y += 26
        gavg = (results['gaze_l'] + results['gaze_r']) / 2.0
        put(frame, f"avg={gavg:.2f}  lo={self.GAZE_LO:.2f}  hi={self.GAZE_HI:.2f}",
            (16, y), self.C_DIM, 0.44);  y += 12
        for gk, lbl in [('gaze_left', '<- LEFT'), ('gaze_right', 'RIGHT ->')]:
            if self._cond_start[gk] is not None:
                frac = 1.0 - results['timers'][gk] / self.ALERT_HOLD_SECS
                self._bar(frame, 16, y, 200, 6, frac, self.C_WARN)
                put(frame, f"{lbl}  {self.ALERT_HOLD_SECS - results['timers'][gk]:.1f}s",
                    (222, y+7), self.C_DIM, 0.40);  y += 16
        y += 8
        cv2.line(frame, (14, y), (panel_w-12, y), (40, 40, 40), 1);  y += 14

        # ── Head ──────────────────────────────────────────────────────────────
        put(frame, "HEAD DIRECTION", (14, y), self.C_DIM, 0.52);  y += 24
        ys = results['yaw_status']
        put(frame, ys.replace('_', ' ').upper(), (16, y),
            self.C_OK if ys == 'straight' else self.C_WARN, 0.70);  y += 26
        yaw_l_delta = self.YAW_L_THRESH - self.YAW_NEUTRAL
        yaw_r_delta = self.YAW_R_THRESH - self.YAW_NEUTRAL
        yaw_l_str = f"{float(yaw_l_delta):.0f}"
        yaw_r_str = f"+{float(yaw_r_delta):.0f}"
        yaw_val   = f"{float(results['yaw']):+.1f}"
        put(frame, f"yaw {yaw_val}deg  thr L={yaw_l_str} R={yaw_r_str}",
            (16, y), self.C_DIM, 0.44);  y += 12
        for hk, lbl in [('head_left', '<- LEFT'), ('head_right', 'RIGHT ->')]:
            if self._cond_start[hk] is not None:
                frac = 1.0 - results['timers'][hk] / self.ALERT_HOLD_SECS
                self._bar(frame, 16, y, 200, 6, frac, self.C_WARN)
                put(frame, f"{lbl}  {self.ALERT_HOLD_SECS - results['timers'][hk]:.1f}s",
                    (222, y+7), self.C_DIM, 0.40);  y += 16
        y += 14
        cv2.line(frame, (14, y), (panel_w-12, y), (40, 40, 40), 1);  y += 16

        # ── Status ────────────────────────────────────────────────────────────
        if results['alerts']:
            for alert in results['alerts']:
                put(frame, f">> {alert}", (16, y), self.C_ALERT, 0.62, 2);  y += 28
        else:
            put(frame, "ATTENTIVE", (16, y), self.C_OK, 0.72, 2)
        put(frame, "R = recalibrate", (14, fh-16), self.C_DIM, 0.44)

    # ─────────────────────────────────────────────────────────────────────────
    # MOUSE
    # ─────────────────────────────────────────────────────────────────────────
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self._btn_rect:
            x1, y1, x2, y2 = self._btn_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.clean_mode = not self.clean_mode

    # ─────────────────────────────────────────────────────────────────────────
    # RUN LOOP
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        # ── Bigger window: request 1280x720 ───────────────────────────────────
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        win = "Driver Distraction Detection"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)   # resizable window
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, self._on_mouse)
        print("\nDriver Distraction Detector ready.")
        print("  Q = quit   S = snapshot   D = toggle clean   R = recalibrate\n")

        last_face   = None
        frame_count = 0
        DETECT_EVERY = 4

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1

            if frame_count % DETECT_EVERY == 0 or last_face is None:
                small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
                faces = self.detector(small, 0)
                if faces:
                    f = max(faces, key=lambda r: r.width() * r.height())
                    last_face = dlib.rectangle(
                        f.left()*2, f.top()*2, f.right()*2, f.bottom()*2)
                else:
                    last_face = None

            if last_face is None:
                self._put(frame, "No face detected", (20, 60), self.C_ALERT, 1.0, 2)
                self.draw_toggle_button(frame)
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('d'): self.clean_mode = not self.clean_mode
                elif key == ord('r'): self.reset_calibration()
                continue

            shape     = self.predictor(gray, last_face)
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            # ── Calibration wizard ────────────────────────────────────────────
            if not self.calibrated:
                # Show intro screen until user presses SPACE
                if not self._intro_shown:
                    self.draw_intro_screen(frame)
                    self.draw_toggle_button(frame)
                    cv2.imshow(win, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): break
                    elif key == ord('r'): self.reset_calibration()
                    elif key == ord(' '): self._intro_shown = True
                    continue

                if self._phase_idx < len(PHASES):
                    if self._done_shown_at is None:
                        self._calib_collect(gray, landmarks, frame.shape)
                        if len(self._phase_samples) >= COLLECT_FRAMES:
                            self._calib_finish_phase()
                    else:
                        if (time.time() - self._done_shown_at) * 1000 >= SHOW_DONE_MS:
                            self._calib_advance_phase()

                self.draw_calibration_screen(frame, self._done_shown_at is not None)
                self.draw_toggle_button(frame)
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('r'): self.reset_calibration()
                continue

            # ── Normal detection ──────────────────────────────────────────────
            results = self.analyze(frame, gray, landmarks)

            if not self.clean_mode:
                x1, y1 = last_face.left(), last_face.top()
                cv2.rectangle(frame, (x1, y1),
                              (x1+last_face.width(), y1+last_face.height()),
                              (100, 100, 100), 1)
                self.draw_overlays(frame, landmarks)
                self.draw_hud(frame, results)

            self.draw_alert_pill(frame, results['alerts'])
            self.draw_toggle_button(frame)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"snapshot_{ts}.png", frame)
                print(f"Saved snapshot_{ts}.png")
            elif key == ord('d'): self.clean_mode = not self.clean_mode
            elif key == ord('r'): self.reset_calibration()

        cap.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    paths = [
        'shape_predictor_68_face_landmarks.dat',
        '../shape_predictor_68_face_landmarks.dat',
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'shape_predictor_68_face_landmarks.dat'),
    ]
    p = next((x for x in paths if os.path.exists(x)), None)
    if p is None:
        print("shape_predictor_68_face_landmarks.dat not found.")
        print("Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return
    profile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'driver_profile.json')
    DriverDistractionDetector(p, profile_path=profile).run()


if __name__ == "__main__":
    main()