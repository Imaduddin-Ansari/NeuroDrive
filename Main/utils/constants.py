"""Constants for NeuroDrive application"""
import os
from pathlib import Path

# ── Resolve the NeuroDrive root (the Main/ folder's parent) ──────────────────
# This file lives at  NeuroDrive/Main/utils/constants.py
# So:  __file__ → .../Main/utils/constants.py
#       parent  → .../Main/utils/
#       parent  → .../Main/
#       parent  → .../NeuroDrive/   ← _ROOT
_THIS_FILE = Path(__file__).resolve()
_MAIN_DIR  = _THIS_FILE.parent.parent          # .../NeuroDrive/Main/
_ROOT      = _MAIN_DIR.parent                  # .../NeuroDrive/

def _rel(*parts):
    """Return an absolute path built from the NeuroDrive root."""
    return str(_ROOT.joinpath(*parts))

# UI Colors
COLORS = {
    'bg_dark': '#0a0a0a',
    'bg_medium': '#0f0f0f',
    'bg_light': '#1a1a1a',
    'bg_lighter': '#252525',
    'accent': '#00a8ff',
    'accent_dark': '#0088cc',
    'text_primary': '#ffffff',
    'text_secondary': '#cccccc',
    'text_tertiary': '#888888',
    'text_dim': '#555555',
    'success': '#00ff66',
    'warning': '#ffaa00',
    'error': '#ff0000',
    'error_light': '#ff6666',
    'purple': '#9d4edd',
    'purple_light': '#c77dff',
}

# Video Feed Settings
VIDEO_SETTINGS = {
    'target_fps': 30,
    'max_queue_size': 2,
    'skip_frames': 2,
    'resize_scale': 0.5,
}

# ── Feed indices ──────────────────────────────────────────────────────────────
FEED_LEFT_BLINDSPOT  = 0
FEED_RIGHT_BLINDSPOT = 1
FEED_FRONT_CAMERA    = 2
FEED_DRIVER_CAMERA   = 3

# Backward-compat aliases
FEED_FRONT_TRAFFIC  = FEED_FRONT_CAMERA
FEED_REAR_FCW_LANE  = FEED_FRONT_CAMERA
FEED_BLINDSPOT      = FEED_LEFT_BLINDSPOT
FEED_PRIORITY_RULES = FEED_FRONT_CAMERA

FEED_LABELS = [
    "Left Side Camera  (Blind Spot)",
    "Right Side Camera  (Blind Spot)",
    "Front Camera  (Traffic · FCW · Lane · Rules)",
    "Driver Camera  (Distraction Detection)",
]

FEED_INDICATORS = {
    FEED_LEFT_BLINDSPOT:  "[LEFT BSM]",
    FEED_RIGHT_BLINDSPOT: "[RIGHT BSM]",
    FEED_FRONT_CAMERA:    "[FRONT]",
    FEED_DRIVER_CAMERA:   "[DRIVER]",
}

# ── Camera / Video Sources ────────────────────────────────────────────────────
CAMERA_SOURCES = {
    'left_blindspot':  _rel("BlindSpotMonitoring", "Pictures", "leftblindspot.mp4"),
    'right_blindspot': _rel("BlindSpotMonitoring", "Pictures", "rightblindspot.mp4"),

    'front_traffic_videos': [
        _rel("TrafficSignDetection", "Videos", "selected1.mp4"),
        _rel("TrafficSignDetection", "Videos", "selected2.mp4"),
        _rel("TrafficSignDetection", "Videos", "selected3.mp4"),
        _rel("TrafficSignDetection", "Videos", "selected4.mp4"),
    ],
    'fcw':            _rel("FCW_COMPLETE", "samplevideos", "test2.mov"),
    'lane_departure': _rel("LaneDepartureWarning", "test.mp4"),
    'priority_rules': _rel("PriorityRulesAlert", "TestPriority.mov"),

    # Driver camera — 0 = webcam; set to a file path if using a recording
    'driver_camera': 0,

    # Pedestrian Intent Prediction — uses dedicated pedestrian video
    'pip': _rel("PedestrianIntentPrediction", "samplevideos", "test4.mp4"),

    # Legacy alias
    'front_videos': [
        _rel("TrafficSignDetection", "Videos", "selected1.mp4"),
        _rel("TrafficSignDetection", "Videos", "selected2.mp4"),
        _rel("TrafficSignDetection", "Videos", "selected3.mp4"),
        _rel("TrafficSignDetection", "Videos", "selected4.mp4"),
    ],
}

# ── Traffic Sign Detection ────────────────────────────────────────────────────
TRAFFIC_SIGN_CONFIG = {
    'model_path':        _rel("TrafficSignDetection", "Models", "traffic_sign_model.h5"),
    'class_names_path':  _rel("TrafficSignDetection", "Models", "class_names.npy"),
    'alert_threshold':   0.90,
    'yolo_conf':         0.12,
    'classifier_conf':   0.7,
    'max_history':       3,
    'display_duration':  5.0,
    'use_image_completion':             True,
    'completion_confidence_threshold':  0.75,
    'templates_dir': _rel("TrafficSignDetection", "data", "templates"),
}

# ── FCW Configuration ─────────────────────────────────────────────────────────
FCW_CONFIG = {
    'yolo_weights':    "yolov8n.pt",
    'ego_kmph':        50.0,
    'danger_distance': 10.0,
    'ttc_threshold':   3.0,
}

# ── Lane Departure Configuration ──────────────────────────────────────────────
LANE_DEPARTURE_CONFIG = {
    'camera_cal_path':    _rel("LaneDepartureWarning", "Images", "camera_cal"),
    'deviation_threshold': 0.5,
    'img_size':           (1280, 720),
}

# ── Priority Rules Configuration ──────────────────────────────────────────────
PRIORITY_RULES_CONFIG = {
    'knowledge_base_path': _rel("PriorityBasedRules", "traffic_rules_knowledge_base.json"),
    'show_overlay':        False,
    'conf_threshold':      0.5,
}

# ── Driver Distraction Configuration ─────────────────────────────────────────
# The .dat and .json both live inside  NeuroDrive/DriverDistractionDetection/Main/
_DDD_MAIN = _ROOT / "DriverDistractionDetection" / "Main"

DRIVER_DISTRACTION_CONFIG = {
    'predictor_path': str(_DDD_MAIN / 'shape_predictor_68_face_landmarks.dat'),
    'profile_path':   str(_DDD_MAIN / 'driver_profile.json'),
}

# ── LLM Risk Explanation Configuration ───────────────────────────────────────
LLM_RISK_CONFIG = {
    # Inference backend: "ollama" or "llamacpp"
    'backend':  "ollama",

    # Base URL of the local model server
    # Ollama default:    http://localhost:11434
    # llama.cpp default: http://localhost:8080
    'base_url': "http://localhost:11434",

    # Model name passed to Ollama (ignored for llama.cpp)
    'model':    "llama3.2",

    # Minimum seconds between consecutive LLM calls.
    # Keeps the model from being hammered during sustained alerts.
    'cooldown': 12.0,

    # HTTP timeout per request (seconds).
    # A 3 B model on CPU typically responds in 1–5 s; 8 s gives headroom.
    'timeout':  8.0,

    # Speak explanations aloud via pyttsx3
    'tts_enabled': True,

    # Write a JSON-lines event log to LLMRiskExplanation/logs/
    'log_enabled': True,
}

# ── UI Dimensions ─────────────────────────────────────────────────────────────
UI_DIMENSIONS = {
    'window_width':      1280,
    'window_height':     800,
    'top_bar_height':    60,
    'alert_panel_height': 180,
    'feed_header_height': 35,
}

# ── Alert Settings ────────────────────────────────────────────────────────────
ALERT_SETTINGS = {
    'max_lines':                    6,
    'fcw_alert_interval':           1.0,
    'blindspot_alert_interval':     2.0,
    'lane_departure_alert_interval': 2.0,
    'priority_rules_alert_interval': 0.5,
}

# ── Front-camera cycling durations (seconds per segment) ─────────────────────
FRONT_CAMERA_CONFIG = {
    'traffic_sign_duration': 45.0,
    'fcw_duration':          30.0,
    'lane_duration':         30.0,
    'priority_duration':     30.0,
    # pip is NOT in the cycle — it overlays on ALL segments
    'cycle_order': ['traffic_sign', 'fcw', 'lane', 'priority_rules'],
}

# ── Alternating Video Settings (legacy) ──────────────────────────────────────
ALTERNATING_CONFIG = {
    'fcw_lane_switch_interval':   30.0,
    'blindspot_switch_interval':  30.0,
}

# ── Driving Style Feedback Configuration ─────────────────────────────────────
DSF_CONFIG = {
    'config_path':    str(_MAIN_DIR / 'dsf_config.yaml'),
    'speed_kmh':      60.0,
    # Set to a video file path for demo; None = uses synthetic sensor data only
    'video_source':   None,
}

# ── Pedestrian Intent Prediction Configuration ────────────────────────────────
PIP_CONFIG = {
    'model_path':        str(_ROOT / "PedestrianIntentPrediction" / "PIP" / "models" / "intent_model.pth"),
    'vehicle_speed_kmh': 30.0,
}
