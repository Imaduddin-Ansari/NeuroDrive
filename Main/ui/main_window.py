"""
main_window.py — NeuroDrive UI

Performance notes
─────────────────
• A single `_tick()` loop (16 ms) replaces four competing `after()` chains.
  This halves Tkinter event-queue pressure and makes all four feeds smooth.
• Frame conversion (BGR→RGB + resize) happens once per tick, only when a
  new frame is actually available from the processor queue.
• Alerts (top-bar indicators + alert panel) are updated every 10th tick
  (~6 Hz) — fast enough to feel live, cheap enough not to stall the UI.
• Front-camera alerts are only shown for the currently-active segment.
• LLM explanations are polled every tick; the explainer worker runs in its
  own daemon thread and never blocks the UI.
• Overtake Assistance button lives on the top bar; side is picked via the
  existing indicator buttons.
"""

import tkinter as tk
import cv2
import time
import os
from pathlib import Path
from PIL import Image, ImageTk

from config import Config
from utils.constants import (
    COLORS, UI_DIMENSIONS, ALERT_SETTINGS,
    FEED_LEFT_BLINDSPOT, FEED_RIGHT_BLINDSPOT,
    FEED_FRONT_CAMERA, FEED_DRIVER_CAMERA,
    CAMERA_SOURCES, TRAFFIC_SIGN_CONFIG, FCW_CONFIG,
    LANE_DEPARTURE_CONFIG, PRIORITY_RULES_CONFIG,
    DRIVER_DISTRACTION_CONFIG, FRONT_CAMERA_CONFIG,
    LLM_RISK_CONFIG, DSF_CONFIG, PIP_CONFIG,
)

from processors.fcw_processor import ForwardCollisionProcessor
from processors.lane_departure_processor import LaneDepartureProcessor
from processors.traffic_sign_processor import TrafficSignProcessor
from processors.blindspot_processor import BlindSpotProcessor
from processors.priority_rules_processor import PriorityRulesProcessor
from processors.driver_distraction_processor import DriverDistractionProcessor
from processors.overtake_processor import OvertakeProcessor
from processors.dsf_processor import DrivingStyleFeedbackProcessor
from processors.pip_processor import PedestrianIntentProcessor

# LLM Risk Explanation module
from LLMRiskExplanation import LLMRiskExplainer, RiskSnapshot
from LLMRiskExplanation.tts_speaker import TTSSpeaker

from ui.loading_screen import LoadingScreen
from ui.settings_window import SettingsWindow
from ui.components.top_bar import TopBar
from ui.components.video_feed import VideoFeedGrid
from ui.components.alert_panel import AlertPanel
from ui.calibration_overlay import CalibrationOverlay, NeuroDrivePopup

# Overtake module
from OvertakeAssistance import OvertakeState, OvertakeSide


class NeuroDriveUI:
    """Main UI window — single-tick update loop for smooth multi-feed display."""

    # ── Timing ────────────────────────────────────────────────────────────────
    _TICK_MS         = 50    # 20 fps — imperceptible on dashcam UI, saves ~67% PIL CPU
    _INDICATOR_EVERY = 6     # update indicators every N ticks (~2.4 Hz at 20fps)
    _CALIB_POLL_MS   = 100

    def __init__(self, root):
        self.root = root
        self.root.title("NeuroDrive — Advanced Driver Assistance System")
        self.root.geometry(
            f"{UI_DIMENSIONS['window_width']}x{UI_DIMENSIONS['window_height']}"
        )
        self.root.configure(bg=COLORS['bg_dark'])

        self.config = Config()

        # ── Processors ────────────────────────────────────────────────────────
        self.left_bsp_processor       = None
        self.right_bsp_processor      = None
        self.traffic_sign_processor   = None
        self.fcw_processor            = None
        self.lane_processor           = None
        self.priority_rules_processor = None
        self.driver_processor         = None
        self.dsf_processor            = None   # Module 9: Driving Style Feedback
        self.pip_processor            = None   # Module 11: Pedestrian Intent Prediction

        # ── Overtake Processor ────────────────────────────────────────────────
        self.overtake_processor = OvertakeProcessor()

        # ── Fallback TTS for overtake verdicts when LLM explainer is disabled ──
        # Only used when LLM-Based Risk Explanation is off in settings.
        self._overtake_fallback_speaker: TTSSpeaker = None

        # ── LLM Risk Explainer ────────────────────────────────────────────────
        self.llm_explainer: LLMRiskExplainer = None

        # ── Front-camera cycling ──────────────────────────────────────────────
        self._cycle_order    = FRONT_CAMERA_CONFIG['cycle_order']
        self._cycle_index    = 0
        self._cycle_start    = time.time()
        self._cycle_durations = {
            'traffic_sign':   FRONT_CAMERA_CONFIG['traffic_sign_duration'],
            'fcw':            FRONT_CAMERA_CONFIG['fcw_duration'],
            'lane':           FRONT_CAMERA_CONFIG['lane_duration'],
            'priority_rules': FRONT_CAMERA_CONFIG['priority_duration'],
        }

        # ── Traffic sign history ──────────────────────────────────────────────
        self.traffic_sign_history  = []
        self.max_sign_history      = TRAFFIC_SIGN_CONFIG['max_history']
        self.sign_display_duration = TRAFFIC_SIGN_CONFIG['display_duration']

        # ── UI handles ────────────────────────────────────────────────────────
        self.top_bar     = None
        self.video_grid  = None
        self.alert_panel = None

        # ── Tick loop state ───────────────────────────────────────────────────
        self.is_updating  = False
        self._tick_id     = None
        self._tick_count  = 0

        # ── Maximize overlay ──────────────────────────────────────────────────
        self._max_window    = None
        self._max_feed_idx  = None
        self._max_tick_id   = None

        # ── Startup errors ────────────────────────────────────────────────────
        self._startup_errors = []

        # ── PIP alert cooldown tracker ────────────────────────────────────────
        self.displayed_pip_alerts: dict = {}  # ped_id -> last shown timestamp

        # ── Overtake state mirror (for UI feedback) ───────────────────────────
        self._last_overtake_state = OvertakeState.IDLE

        # ── LLM consecutive-alert gate ────────────────────────────────────────
        # The LLM is only submitted when the same alert condition has been
        # active for at least this many consecutive indicator ticks (~6 Hz),
        # i.e. roughly 3 × (1/6 Hz) ≈ 0.5 s of sustained alerting.
        # Overtake verdicts bypass this gate entirely (handled separately).
        self._llm_required_consecutive = 3
        self._llm_consecutive_alerts   = 0   # ticks where active_alert_count >= 2

        # ── Boot ──────────────────────────────────────────────────────────────
        self.loading_screen = LoadingScreen(self.root)
        self.loading_screen.show()
        self.root.after(500, self._start_driver_and_gate)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — start driver processor, gate on calibration
    # ══════════════════════════════════════════════════════════════════════════

    def _start_driver_and_gate(self):
        self.loading_screen.update_message("Starting driver camera…")
        self.root.update_idletasks()

        driver_src = CAMERA_SOURCES.get('driver_camera', 0)
        try:
            self.driver_processor = DriverDistractionProcessor(
                video_source=driver_src,
                predictor_path=DRIVER_DISTRACTION_CONFIG['predictor_path'],
                profile_path=DRIVER_DISTRACTION_CONFIG['profile_path'],
            )
            self.driver_processor.detection_enabled = self.config.get(
                "Driver Distraction Detection", True)
            self.driver_processor.start()
        except Exception as e:
            self._startup_errors.append(f"Driver Distraction Detection failed:\n{e}")
            self.root.after(1500, self._proceed_to_main)
            return

        profile_path = DRIVER_DISTRACTION_CONFIG['profile_path']
        if os.path.exists(profile_path):
            self.loading_screen.update_message("Profile found — loading system…")
            self.root.after(1500, self._proceed_to_main)
        else:
            self.loading_screen.update_message(
                "No driver profile found.\n"
                "Please complete the calibration wizard…"
            )
            calib = CalibrationOverlay(
                self.root,
                self.driver_processor,
                on_complete=self._on_calibration_done,
            )
            calib.show()

    def _on_calibration_done(self):
        self.loading_screen.update_message("Calibration complete — loading system…")
        self.root.after(1000, self._proceed_to_main)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — main screen
    # ══════════════════════════════════════════════════════════════════════════

    def _proceed_to_main(self):
        self.loading_screen.hide()
        self._show_main_screen()

    def _show_main_screen(self):
        self.main_frame = tk.Frame(self.root, bg=COLORS['bg_medium'])
        self.main_frame.pack(fill='both', expand=True)

        self.top_bar = TopBar(
            self.main_frame,
            self.show_settings,
            indicator_callback=self._on_indicator_change,
            overtake_callback=self._on_overtake_button,   # NEW
        )
        self.top_bar.pack(fill='x', side='top')

        self.alert_panel = AlertPanel(self.main_frame)
        self.alert_panel.pack(fill='x', side='bottom', padx=15, pady=(0, 15))
        self.alert_panel.pack_propagate(False)

        self.video_grid = VideoFeedGrid(self.main_frame, self.maximize_feed)
        self.video_grid.pack(fill='both', expand=True, padx=15, pady=15)

        self._start_all_modules()
        self._start_tick()

        self.root.after(600, self._show_startup_errors)

    def _show_startup_errors(self):
        if self.driver_processor and self.driver_processor.startup_error:
            err = self.driver_processor.startup_error
            if err not in self._startup_errors:
                self._startup_errors.append(err)
        if self._startup_errors:
            NeuroDrivePopup(
                self.root,
                title="NeuroDrive — Startup Warnings",
                messages=self._startup_errors,
                level='warning',
            )
            self._startup_errors.clear()

    # ── Indicator change: also feeds overtake side selection ─────────────────

    def _on_indicator_change(self, left_active, right_active):
        # Lane departure
        if self.lane_processor and self.lane_processor.is_running:
            self.lane_processor.set_indicators(left=left_active, right=right_active)

        # Overtake side selection — only when processor is WAITING
        ot_state = self.overtake_processor.state
        if ot_state == OvertakeState.WAITING:
            if left_active:
                self.overtake_processor.set_side(OvertakeSide.LEFT)
            elif right_active:
                self.overtake_processor.set_side(OvertakeSide.RIGHT)

    # ── Overtake button ───────────────────────────────────────────────────────

    def _on_overtake_button(self):
        self.overtake_processor.request_overtake()
        # Immediately reflect new state on button
        self._sync_overtake_ui()

    def _sync_overtake_ui(self):
        """Push current overtake state to the top bar button."""
        if not self.top_bar:
            return
        state = self.overtake_processor.state
        result = self.overtake_processor.latest_result

        state_str = state.value  # 'idle', 'waiting', 'checking', 'safe', 'caution', 'unsafe'

        sub = ""
        if state == OvertakeState.WAITING:
            sub = "Press ◀L or R▶ to choose side"
        elif state == OvertakeState.CHECKING:
            side = self.overtake_processor.active_side
            sub  = f"Analysing {side.value} side…" if side else "Analysing…"
        elif result is not None:
            side_txt = result.side.value.upper() if result.side else ""
            if state == OvertakeState.SAFE:
                sub = f"{side_txt} — path clear"
            elif state == OvertakeState.CAUTION:
                sub = f"{side_txt} — proceed carefully"
            elif state == OvertakeState.UNSAFE:
                sub = f"{side_txt} — DO NOT overtake"

        self.top_bar.update_overtake_status(state_str, sub)

    # ── Overtake result callback (fires from background thread) ──────────────

    def _on_overtake_result(self, result):
        """Called by OvertakeProcessor when analysis completes."""
        # All UI ops must be scheduled on the main thread
        self.root.after(0, self._handle_overtake_result, result)

    def _handle_overtake_result(self, result):
        """Main-thread handler — updates UI, speaks verdict, fires LLM."""
        self._sync_overtake_ui()

        side_txt = result.side.value if result.side else "unknown side"
        side_upper = side_txt.upper()

        # Post to alert panel
        if self.alert_panel:
            verdict  = result.state.value.upper()
            emoji    = ("✅" if result.state == OvertakeState.SAFE else
                        "⚠️" if result.state == OvertakeState.CAUTION else "🚫")
            msg_tag  = ("low" if result.state == OvertakeState.SAFE else
                        "high" if result.state == OvertakeState.CAUTION else "critical")
            self.alert_panel._append_line(
                f"{emoji} OVERTAKE ({side_upper}): {verdict} — {result.reason}",
                msg_tag
            )

        # ── Single unified voice for overtake verdict ─────────────────────────
        # If LLM explainer is active, submit to it — it generates and speaks a
        # single explanation. Nothing else speaks, preventing double-audio.
        # If LLM is disabled, a short pre-written message is spoken instead.
        if self.llm_explainer:
            snap = self._build_base_snapshot(time.time())
            snap.overtake_requested        = True
            snap.overtake_side             = result.side.value if result.side else None
            snap.overtake_verdict          = result.state.value
            snap.overtake_reason           = result.reason
            snap.overtake_blindspot_clear  = result.blindspot_clear
            snap.overtake_front_clear      = result.front_clear
            snap.overtake_front_dist_m     = result.front_vehicle_dist_m
            snap.overtake_gap_seconds      = result.safe_gap_seconds
            snap.overtake_approaching_rear = result.approaching_from_behind
            snap.overtake_notes            = result.notes
            self.llm_explainer.submit(snap)
            print(f"[Overtake] Verdict submitted to LLM — LLM will speak.")
        else:
            # LLM disabled — speak a concise pre-written verdict directly.
            if result.state == OvertakeState.SAFE:
                tts_msg = f"The {side_txt} side is clear. You may overtake."
            elif result.state == OvertakeState.CAUTION:
                tts_msg = f"Caution on the {side_txt} side. Wait for a larger gap."
            else:
                tts_msg = f"Do not overtake on the {side_txt}. The lane is not clear."
            print(f"[OvertakeTTS] {tts_msg}")
            if self._overtake_fallback_speaker is None:
                self._overtake_fallback_speaker = TTSSpeaker(enabled=True)
            self._overtake_fallback_speaker.speak(tts_msg)


    # ══════════════════════════════════════════════════════════════════════════
    # Module startup
    # ══════════════════════════════════════════════════════════════════════════

    def _safe_start(self, label, factory):
        try:
            return factory()
        except Exception as e:
            self._startup_errors.append(f"{label}:\n{e}")
            print(f"[Startup] ✗ {label}: {e}")
            return None

    def _start_all_modules(self):
        cfg = self.config

        # Feed 0 — Left blind spot
        src = CAMERA_SOURCES['left_blindspot']
        if os.path.exists(src):
            def _mk():
                p = BlindSpotProcessor(src, side='left')
                p.detection_enabled = cfg.get("Blind Spot Monitoring", True)
                p.start(); return p
            self.left_bsp_processor = self._safe_start("Left Blind Spot", _mk)
            if self.left_bsp_processor:
                print("✓ Left Blind Spot started")

        # Feed 1 — Right blind spot
        src = CAMERA_SOURCES['right_blindspot']
        if os.path.exists(src):
            def _mk():
                p = BlindSpotProcessor(src, side='right')
                p.detection_enabled = cfg.get("Blind Spot Monitoring", True)
                p.start(); return p
            self.right_bsp_processor = self._safe_start("Right Blind Spot", _mk)
            if self.right_bsp_processor:
                print("✓ Right Blind Spot started")

        # Feed 2a — Traffic signs
        existing = [v for v in CAMERA_SOURCES['front_traffic_videos'] if os.path.exists(v)]
        if existing:
            def _mk():
                p = TrafficSignProcessor(
                    video_source=existing,
                    model_path=TRAFFIC_SIGN_CONFIG['model_path'],
                    class_names_path=TRAFFIC_SIGN_CONFIG['class_names_path'],
                    alert_threshold=TRAFFIC_SIGN_CONFIG['alert_threshold'],
                    use_image_completion=TRAFFIC_SIGN_CONFIG['use_image_completion'],
                    completion_threshold=TRAFFIC_SIGN_CONFIG['completion_confidence_threshold'],
                    templates_dir=TRAFFIC_SIGN_CONFIG['templates_dir'],
                )
                p.detection_enabled = cfg.get("Traffic Sign Detection", True)
                p.start(); return p
            self.traffic_sign_processor = self._safe_start("Traffic Sign Detection", _mk)
            if self.traffic_sign_processor:
                print(f"✓ Traffic Sign Detection started ({len(existing)} videos)")

        # Feed 2b — FCW
        src = CAMERA_SOURCES['fcw']
        if os.path.exists(src):
            def _mk():
                p = ForwardCollisionProcessor(
                    video_path=src,
                    yolo_weights=FCW_CONFIG['yolo_weights'],
                    ego_kmph=FCW_CONFIG['ego_kmph'],
                )
                p.detection_enabled = cfg.get("Forward Collision Warning", True)
                p.start(); return p
            self.fcw_processor = self._safe_start("Forward Collision Warning", _mk)
            if self.fcw_processor:
                print("✓ Forward Collision Warning started")

        # Feed 2c — Lane departure
        src = CAMERA_SOURCES['lane_departure']
        if os.path.exists(src):
            def _mk():
                p = LaneDepartureProcessor(video_source=src)
                p.detection_enabled = cfg.get("Lane Departure Warning", True)
                p.start(); return p
            self.lane_processor = self._safe_start("Lane Departure Warning", _mk)
            if self.lane_processor:
                print("✓ Lane Departure Warning started")

        # Feed 2d — Priority rules
        src = CAMERA_SOURCES['priority_rules']
        if os.path.exists(src):
            def _mk():
                p = PriorityRulesProcessor(video_source=src)
                p.detection_enabled = cfg.get("Priority-Based Rules Alert", True)
                p.start(); return p
            self.priority_rules_processor = self._safe_start("Priority Rules", _mk)
            if self.priority_rules_processor:
                print("✓ Priority Rules Detection started")

        # Feed 3 — Driver (already running from Phase 1)
        if self.driver_processor:
            self.driver_processor.detection_enabled = cfg.get(
                "Driver Distraction Detection", True)
            print("✓ Driver Distraction (already running)")

        # ── Module 9: Driving Style Feedback ─────────────────────────────────
        dsf_on = cfg.get("Driving Style Feedback", True)
        if dsf_on:
            def _mk_dsf():
                p = DrivingStyleFeedbackProcessor(
                    config_path=DSF_CONFIG['config_path'],
                    speed_kmh=DSF_CONFIG['speed_kmh'],
                )
                p.detection_enabled = True
                p.start()
                return p
            self.dsf_processor = self._safe_start("Driving Style Feedback", _mk_dsf)
            if self.dsf_processor:
                print("✓ Driving Style Feedback started")
        else:
            print("— Driving Style Feedback disabled in config")

        # ── Module 11: Pedestrian Intent Prediction ───────────────────────────
        pip_on = cfg.get("Pedestrian Intent Prediction", True)
        if pip_on:
            def _mk_pip():
                p = PedestrianIntentProcessor(
                    model_path=PIP_CONFIG.get('model_path'),
                    vehicle_speed_kmh=PIP_CONFIG['vehicle_speed_kmh'],
                )
                p.detection_enabled = True
                p.start()
                return p
            self.pip_processor = self._safe_start("Pedestrian Intent Prediction", _mk_pip)
            if self.pip_processor:
                print("✓ Pedestrian Intent Prediction started (overlay mode)")
        else:
            print("— Pedestrian Intent Prediction disabled in config")

        # ── LLM Risk Explainer ────────────────────────────────────────────────
        llm_on = cfg.get("LLM-Based Risk Explanation", True)
        if llm_on:
            try:
                self.llm_explainer = LLMRiskExplainer(
                    backend=     LLM_RISK_CONFIG['backend'],
                    base_url=    LLM_RISK_CONFIG['base_url'],
                    model=       LLM_RISK_CONFIG['model'],
                    cooldown=    LLM_RISK_CONFIG['cooldown'],
                    timeout=     LLM_RISK_CONFIG['timeout'],
                    tts_enabled= LLM_RISK_CONFIG['tts_enabled'],
                    log_enabled= LLM_RISK_CONFIG['log_enabled'],
                )
                print("✓ LLM Risk Explainer started")
            except Exception as e:
                self._startup_errors.append(f"LLM Risk Explainer:\n{e}")
                print(f"[Startup] ✗ LLM Risk Explainer: {e}")
        else:
            print("— LLM Risk Explainer disabled in config")

        # ── Wire Overtake Processor ───────────────────────────────────────────
        ot_on = cfg.get("Overtake Assistance", True)
        if ot_on:
            self.overtake_processor.attach_processors(
                left_bsp        = self.left_bsp_processor,
                right_bsp       = self.right_bsp_processor,
                fcw             = self.fcw_processor,
                get_front_frame_fn = self._get_latest_front_frame,
                ego_kmph_fn     = self._get_ego_speed,
            )
            self.overtake_processor.on_result_ready = self._on_overtake_result
            # Inject YOLO from FCW processor if available
            if self.fcw_processor and self.fcw_processor.yolo:
                self.overtake_processor.inject_yolo(self.fcw_processor.yolo)
            print("✓ Overtake Assistance wired")
        else:
            print("— Overtake Assistance disabled in config")

    # ── Helpers for overtake processor ───────────────────────────────────────

    def _get_latest_front_frame(self):
        """Return the most recent front-camera frame (BGR ndarray) or None."""
        seg = self._cycle_order[self._cycle_index]
        frame = None
        if seg == 'fcw' and self.fcw_processor:
            frame = self.fcw_processor.get_processed_frame()
        elif seg == 'traffic_sign' and self.traffic_sign_processor:
            frame = self.traffic_sign_processor.get_processed_frame()
        # Fallback: try all front processors
        if frame is None and self.fcw_processor:
            frame = self.fcw_processor.get_processed_frame()
        return frame

    def _get_ego_speed(self) -> float:
        if self.fcw_processor and hasattr(self.fcw_processor, 'ego_kmph'):
            return self.fcw_processor.ego_kmph
        return FCW_CONFIG.get('ego_kmph', 50.0)

    # ══════════════════════════════════════════════════════════════════════════
    # Unified tick loop
    # ══════════════════════════════════════════════════════════════════════════

    def _start_tick(self):
        self.is_updating = True
        self._tick()

    @staticmethod
    def _drain_latest(processor):
        import queue as _q
        latest = None
        while True:
            try:
                f = processor.frame_queue.get_nowait()
                if f is not None:
                    latest = f
            except _q.Empty:
                break
        return latest

    def _tick(self):
        if not self.is_updating:
            return

        self._tick_count += 1

        # ── Advance front-camera segment ──────────────────────────────────────
        now     = time.time()
        segment = self._cycle_order[self._cycle_index]
        if now - self._cycle_start >= self._cycle_durations.get(segment, 30.0):
            self._cycle_index = (self._cycle_index + 1) % len(self._cycle_order)
            self._cycle_start = now
            segment = self._cycle_order[self._cycle_index]
            print(f"[FrontCamera] → {segment}")

        # ── Feed 0 : Left blind spot ──────────────────────────────────────────
        if self.left_bsp_processor and self.left_bsp_processor.is_running:
            f = self._drain_latest(self.left_bsp_processor)
            if f is not None:
                self.video_grid.update_feed(FEED_LEFT_BLINDSPOT, f)

        # ── Feed 1 : Right blind spot ─────────────────────────────────────────
        if self.right_bsp_processor and self.right_bsp_processor.is_running:
            f = self._drain_latest(self.right_bsp_processor)
            if f is not None:
                self.video_grid.update_feed(FEED_RIGHT_BLINDSPOT, f)

        # ── Feed 2 : Active front-camera segment ──────────────────────────────
        frame2 = None
        if segment == 'traffic_sign' and self.traffic_sign_processor:
            frame2 = self._drain_latest(self.traffic_sign_processor)
        elif segment == 'fcw' and self.fcw_processor:
            frame2 = self._drain_latest(self.fcw_processor)
        elif segment == 'lane' and self.lane_processor:
            frame2 = self._drain_latest(self.lane_processor)
        elif segment == 'priority_rules' and self.priority_rules_processor:
            frame2 = self._drain_latest(self.priority_rules_processor)

        # ── PIP overlay: push raw frame in, get annotated frame back ──────────
        # Works on ALL front camera segments — not just one video
        pip_on = self.config.get("Pedestrian Intent Prediction", True)
        if pip_on and self.pip_processor and self.pip_processor.is_running:
            if frame2 is not None:
                self.pip_processor.push_frame(frame2)
            # Returns latest annotated frame OR last cached one — never None after first result
            pip_frame = self.pip_processor.get_annotated_frame()
            if pip_frame is not None:
                frame2 = pip_frame   # replace with annotated version

        if frame2 is not None:
            self.video_grid.update_feed(FEED_FRONT_CAMERA, frame2)

        # ── Feed 3 : Driver camera ────────────────────────────────────────────
        if self.driver_processor and self.driver_processor.is_running:
            f = self._drain_latest(self.driver_processor)
            if f is not None:
                self.video_grid.update_feed(FEED_DRIVER_CAMERA, f)

        # ── Poll LLM explainer ────────────────────────────────────────────────
        if self.llm_explainer and self.alert_panel:
            explanation = self.llm_explainer.poll()
            if explanation:
                self.alert_panel.add_llm_explanation(explanation, now)

        # ── Update DSF bottom panel ───────────────────────────────────────────
        if self.dsf_processor and self.alert_panel:
            self.alert_panel.update_dsf(self.dsf_processor.panel_data)

        # ── Tick the overtake processor (timeout / expiry) ────────────────────
        new_ot_state = self.overtake_processor.tick()
        if new_ot_state != self._last_overtake_state:
            self._last_overtake_state = new_ot_state
            self._sync_overtake_ui()

        # Update ego speed reference in overtake processor
        self.overtake_processor.update_ego_speed()

        # ── Indicators & alerts (every _INDICATOR_EVERY ticks) ────────────────
        if self._tick_count % self._INDICATOR_EVERY == 0:
            self._update_indicators(segment, now)
            if self.llm_explainer:
                self._submit_risk_snapshot(now)

        self._tick_id = self.root.after(self._TICK_MS, self._tick)

    # ══════════════════════════════════════════════════════════════════════════
    # LLM snapshot builder
    # ══════════════════════════════════════════════════════════════════════════

    def _build_base_snapshot(self, now: float) -> RiskSnapshot:
        """Build a snapshot with all non-overtake fields populated."""
        snap = RiskSnapshot(timestamp=now)

        if self.fcw_processor and self.fcw_processor.running:
            snap.fcw_critical = self.fcw_processor.critical
            snap.ego_speed_kmh = self.fcw_processor.ego_kmph
            if hasattr(self.fcw_processor, 'last_depth_m'):
                snap.fcw_depth_m = self.fcw_processor.last_depth_m
            if hasattr(self.fcw_processor, 'last_ttc_s'):
                snap.fcw_ttc_s = self.fcw_processor.last_ttc_s

        if self.left_bsp_processor and self.left_bsp_processor.is_running:
            snap.left_bsp      = self.left_bsp_processor.detection_status
            snap.bsp_distance_m = self.left_bsp_processor.vehicle_distance
        if self.right_bsp_processor and self.right_bsp_processor.is_running:
            snap.right_bsp = self.right_bsp_processor.detection_status
            if self.right_bsp_processor.vehicle_distance is not None:
                if snap.bsp_distance_m is None:
                    snap.bsp_distance_m = self.right_bsp_processor.vehicle_distance
                else:
                    snap.bsp_distance_m = min(snap.bsp_distance_m,
                                              self.right_bsp_processor.vehicle_distance)

        if self.lane_processor and self.lane_processor.is_running:
            snap.lane_warning    = self.lane_processor.deviation_warning
            snap.lane_direction  = self.lane_processor.direction
            snap.lane_position_m = self.lane_processor.position

        if (self.traffic_sign_processor and
                self.traffic_sign_processor.is_running and
                self.traffic_sign_processor.detection_status):
            snap.sign_class      = self.traffic_sign_processor.current_sign
            snap.sign_confidence = self.traffic_sign_processor.sign_confidence

        if self.priority_rules_processor and self.priority_rules_processor.is_running:
            snap.active_rules = self.priority_rules_processor.active_rules

        if self.driver_processor and self.driver_processor.is_running:
            snap.driver_distracted = self.driver_processor.distracted
            snap.driver_alerts     = self.driver_processor.alerts
            snap.eyes_open         = self.driver_processor.eyes_open
            snap.gaze              = self.driver_processor.gaze
            if hasattr(self.driver_processor, 'yaw_status'):
                snap.yaw_status = self.driver_processor.yaw_status

        # ── Module 9: DSF — forward ALL module alert states ──────────────────
        if self.dsf_processor and self.dsf_processor.is_running:
            self.dsf_processor.update_alert_state({
                "fcw_alert":          bool(snap.fcw_critical),
                "lane_alert":         bool(snap.lane_warning),
                "bsp_alert":          bool(snap.left_bsp or snap.right_bsp),
                "sign_alert":         bool(snap.sign_class),
                "priority_alert":     bool(snap.active_rules),
                "drowsiness_alert":   bool(snap.driver_distracted),
                "distraction_alert":  bool(snap.driver_distracted),
                "pip_alert":          bool(self.pip_processor and
                                           self.pip_processor.detection_status),            })

        # ── Module 11: PIP ────────────────────────────────────────────────────
        if self.pip_processor and self.pip_processor.is_running:
            snap.pedestrian_alert = self.pip_processor.detection_status

        return snap

    def _submit_risk_snapshot(self, now: float) -> None:
        """
        Build a snapshot and submit it to the LLM only when ALL of:
          • 3 or more distinct alert modules are active simultaneously, AND
          • that condition has been sustained for at least
            _llm_required_consecutive indicator ticks (~6 Hz ≈ 0.5 s).
        Overtake verdicts bypass this gate entirely — handled in
        _handle_overtake_result.
        """
        snap = self._build_base_snapshot(now)

        # Require 3 distinct alert sources (not just 2) before starting
        # the consecutive-tick counter.  This prevents a single module that
        # happens to trigger two internal flags from firing the LLM alone.
        alert_count = snap.active_alert_count()

        if alert_count >= 3:
            self._llm_consecutive_alerts += 1
        else:
            # Condition cleared or insufficient — reset so it must sustain again
            self._llm_consecutive_alerts = 0

        # Only submit once the 3-alert condition has been continuous for the
        # required number of ticks.  The LLM's own cooldown prevents
        # re-firing immediately after.
        if self._llm_consecutive_alerts >= self._llm_required_consecutive:
            self.llm_explainer.submit(snap)

    # ══════════════════════════════════════════════════════════════════════════
    # Indicator / alert updates  (called at ~6 Hz from _tick)
    # ══════════════════════════════════════════════════════════════════════════

    def _update_indicators(self, segment, current_time):
        if not self.top_bar:
            return

        fcw_on = self.config.get("Forward Collision Warning", True)
        if not fcw_on:
            self.top_bar.update_fcw_status(label_override="FCW: OFF")
        elif segment == 'fcw' and self.fcw_processor and self.fcw_processor.running:
            self.top_bar.update_fcw_status(critical=self.fcw_processor.critical)
        else:
            self.top_bar.update_fcw_status(offline=True)

        lane_on = self.config.get("Lane Departure Warning", True)
        if not lane_on:
            self.top_bar.update_lane_status(label_override="LANE: OFF")
        elif segment == 'lane' and self.lane_processor and self.lane_processor.is_running:
            self.top_bar.update_lane_status(
                warning=self.lane_processor.deviation_warning,
                direction=self.lane_processor.direction,
            )
        else:
            self.top_bar.update_lane_status(offline=True)

        ts_on = self.config.get("Traffic Sign Detection", True)
        if not ts_on:
            self.top_bar.update_traffic_signs([], current_time, label_override="SIGNS: OFF")
        elif segment == 'traffic_sign':
            self._refresh_sign_history(current_time)
            self.top_bar.update_traffic_signs(self.traffic_sign_history, current_time)
        else:
            self.top_bar.update_traffic_signs([], current_time)

        bsp_on = self.config.get("Blind Spot Monitoring", True)
        if not bsp_on:
            self.top_bar.update_blindspot_status(detected=False, label_override="BLIND: OFF")
        else:
            left_det  = bool(self.left_bsp_processor  and
                             self.left_bsp_processor.is_running  and
                             self.left_bsp_processor.detection_status)
            right_det = bool(self.right_bsp_processor and
                             self.right_bsp_processor.is_running and
                             self.right_bsp_processor.detection_status)
            if left_det or right_det:
                side = 'left' if left_det else 'right'
                proc = self.left_bsp_processor if left_det else self.right_bsp_processor
                self.top_bar.update_blindspot_status(
                    detected=True, side=side,
                    count=proc.vehicle_count, distance=proc.vehicle_distance,
                )
            else:
                self.top_bar.update_blindspot_status(detected=False)

        rules_on = self.config.get("Priority-Based Rules Alert", True)
        if not rules_on:
            self.top_bar.update_priority_rules([], label_override="RULES: OFF")
        elif (segment == 'priority_rules' and
              self.priority_rules_processor and
              self.priority_rules_processor.is_running):
            active = self.priority_rules_processor.active_rules
            self.top_bar.update_priority_rules(active)
            for rule in active:
                self.alert_panel.add_priority_rule(rule, current_time)
        else:
            self.top_bar.update_priority_rules([])

        driver_on = self.config.get("Driver Distraction Detection", True)
        if hasattr(self.top_bar, 'update_driver_status'):
            if not driver_on:
                self.top_bar.update_driver_status(
                    distracted=False, alerts=[], label_override="DRIVER: OFF")
            elif self.driver_processor:
                self.top_bar.update_driver_status(
                    distracted=self.driver_processor.distracted,
                    alerts=self.driver_processor.alerts,
                )

        # ── PIP alerts → left alert panel (only high-intent, with cooldown) ────
        pip_on = self.config.get("Pedestrian Intent Prediction", True)
        if pip_on and self.pip_processor and self.pip_processor.is_running:
            for alert_res in self.pip_processor.crossing_alerts:
                intent = alert_res.get("intent_prob", 0.0)
                # Only show in alert panel if intent >= 0.80 (very probable crossing)
                if intent < 0.80:
                    continue
                ped_id = alert_res.get("id", "?")
                dist   = alert_res.get("distance_m", 0)
                # Per-pedestrian cooldown — same ped can't spam more than once per 5s
                cooldown_key = f"pip_{ped_id}"
                last_shown = self.displayed_pip_alerts.get(cooldown_key, 0)
                if current_time - last_shown < 5.0:
                    continue
                self.displayed_pip_alerts[cooldown_key] = current_time
                if self.alert_panel:
                    msg = f"🚨 PEDESTRIAN crossing likely — intent {intent:.0%}, {dist:.1f}m ahead"
                    self.alert_panel._append_line(msg, "critical")

    def _refresh_sign_history(self, current_time):
        if (self.traffic_sign_processor and
                self.traffic_sign_processor.is_running and
                self.traffic_sign_processor.detection_status):
            name = self.traffic_sign_processor.current_sign
            conf = self.traffic_sign_processor.sign_confidence
            found = False
            for entry in self.traffic_sign_history:
                if entry['name'] == name and (current_time - entry['last_seen']) < 1.0:
                    entry['last_seen']  = current_time
                    entry['confidence'] = max(entry['confidence'], conf)
                    found = True
                    break
            if not found:
                self.traffic_sign_history.append({
                    'name': name, 'confidence': conf,
                    'first_seen': current_time, 'last_seen': current_time,
                    'alerted': False,
                })

        self.traffic_sign_history = [
            s for s in self.traffic_sign_history
            if (current_time - s['last_seen']) < self.sign_display_duration
        ]
        self.traffic_sign_history.sort(key=lambda x: x['confidence'], reverse=True)
        self.traffic_sign_history = self.traffic_sign_history[:self.max_sign_history]

    # ══════════════════════════════════════════════════════════════════════════
    # Maximize / fullscreen overlay
    # ══════════════════════════════════════════════════════════════════════════

    def maximize_feed(self, feed_index):
        if self._max_window is not None:
            self._close_maximize()
            return

        self._max_feed_idx = feed_index
        win = tk.Toplevel(self.root)
        win.title("NeuroDrive — Maximized Feed")
        win.configure(bg='black')
        win.attributes('-fullscreen', True)
        win.bind('<Escape>', lambda e: self._close_maximize())
        win.bind('<Button-1>', lambda e: self._close_maximize())
        self._max_window = win

        self._max_label = tk.Label(win, bg='black')
        self._max_label.pack(expand=True, fill='both')
        tk.Label(win, text="Click or press ESC to close",
                 font=("Helvetica", 12), fg='#555555', bg='black').pack(side='bottom', pady=8)

        self._max_tick()

    def _max_tick(self):
        if self._max_window is None or not self._max_window.winfo_exists():
            return
        frame = self._get_latest_frame(self._max_feed_idx)
        if frame is not None:
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sw   = self._max_window.winfo_screenwidth()
            sh   = self._max_window.winfo_screenheight()
            fh, fw = rgb.shape[:2]
            scale  = min(sw / fw, sh / fh)
            nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
            imgtk  = ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(rgb, (nw, nh))))
            self._max_label.imgtk = imgtk
            self._max_label.configure(image=imgtk)
        self._max_tick_id = self._max_window.after(self._TICK_MS, self._max_tick)

    def _close_maximize(self):
        if self._max_tick_id is not None:
            try:
                self._max_window.after_cancel(self._max_tick_id)
            except Exception:
                pass
        if self._max_window is not None and self._max_window.winfo_exists():
            self._max_window.destroy()
        self._max_window   = None
        self._max_feed_idx = None
        self._max_tick_id  = None

    def _get_latest_frame(self, feed_index):
        seg = self._cycle_order[self._cycle_index]
        if feed_index == FEED_LEFT_BLINDSPOT and self.left_bsp_processor:
            return self.left_bsp_processor.get_processed_frame()
        if feed_index == FEED_RIGHT_BLINDSPOT and self.right_bsp_processor:
            return self.right_bsp_processor.get_processed_frame()
        if feed_index == FEED_FRONT_CAMERA:
            if seg == 'traffic_sign' and self.traffic_sign_processor:
                return self.traffic_sign_processor.get_processed_frame()
            if seg == 'fcw' and self.fcw_processor:
                return self.fcw_processor.get_processed_frame()
            if seg == 'lane' and self.lane_processor:
                return self.lane_processor.get_processed_frame()
            if seg == 'priority_rules' and self.priority_rules_processor:
                return self.priority_rules_processor.get_processed_frame()
        if feed_index == FEED_DRIVER_CAMERA and self.driver_processor:
            return self.driver_processor.get_processed_frame()
        return None

    def _get_latest_front_frame(self):
        """Latest front frame for overtake analysis."""
        seg = self._cycle_order[self._cycle_index]
        frame = None
        if seg == 'fcw' and self.fcw_processor:
            frame = self.fcw_processor.get_processed_frame()
        if frame is None and self.fcw_processor:
            frame = self.fcw_processor.get_processed_frame()
        if frame is None and self.traffic_sign_processor:
            frame = self.traffic_sign_processor.get_processed_frame()
        return frame

    # ══════════════════════════════════════════════════════════════════════════
    # Live settings apply
    # ══════════════════════════════════════════════════════════════════════════

    def apply_settings(self):
        cfg = self.config
        pairs = [
            (self.left_bsp_processor,       "Blind Spot Monitoring"),
            (self.right_bsp_processor,      "Blind Spot Monitoring"),
            (self.traffic_sign_processor,   "Traffic Sign Detection"),
            (self.fcw_processor,            "Forward Collision Warning"),
            (self.lane_processor,           "Lane Departure Warning"),
            (self.priority_rules_processor, "Priority-Based Rules Alert"),
            (self.driver_processor,         "Driver Distraction Detection"),
        ]
        for proc, key in pairs:
            if proc is not None:
                proc.detection_enabled = cfg.get(key, True)
        if self.dsf_processor is not None:
            self.dsf_processor.detection_enabled = cfg.get("Driving Style Feedback", True)
        if self.pip_processor is not None:
            self.pip_processor.detection_enabled = cfg.get("Pedestrian Intent Prediction", True)
        print("[Settings] Applied live.")

    def show_settings(self):
        SettingsWindow(self.root, self.config,
                       alert_callback=lambda _: None,
                       apply_callback=self.apply_settings)

    def is_alert_showing(self, text):
        return self.alert_panel.has_message(text) if self.alert_panel else False

    # ══════════════════════════════════════════════════════════════════════════
    # Shutdown
    # ══════════════════════════════════════════════════════════════════════════

    def stop_video_feeds(self):
        self.is_updating = False
        self._close_maximize()

        if self._tick_id is not None:
            try:
                self.root.after_cancel(self._tick_id)
            except Exception:
                pass
        self._tick_id = None

        if self.llm_explainer:
            try:
                self.llm_explainer.stop()
            except Exception:
                pass

        if self._overtake_fallback_speaker is not None:
            try:
                self._overtake_fallback_speaker.stop()
            except Exception:
                pass

        self.overtake_processor.stop()

        for proc in [
            self.left_bsp_processor, self.right_bsp_processor,
            self.traffic_sign_processor, self.fcw_processor,
            self.lane_processor, self.priority_rules_processor,
            self.driver_processor, self.dsf_processor,
        ]:
            if proc:
                try:
                    proc.stop()
                except Exception:
                    pass

        if self.pip_processor:
            try:
                self.pip_processor.stop()
            except Exception:
                pass

    def build_run_summary(self) -> dict:
        """Collect summaries and stats from active processors to build a
        combined run summary dict suitable for SummaryWindow.
        """
        summary = {}

        # DSF summary
        try:
            dsf = self.dsf_processor.get_summary() if self.dsf_processor else {}
        except Exception:
            dsf = {}
        # Also attempt to include the raw DSF text summary (latest file) for full details
        try:
            summaries_dir = Path('summaries')
            if summaries_dir.exists():
                files = sorted(summaries_dir.glob('summary_*.txt'))
                if files:
                    latest = files[-1]
                    with open(latest, 'r') as fh:
                        dsf_raw = fh.read()
                    dsf = dict(dsf) if isinstance(dsf, dict) else {}
                    dsf['raw_text'] = dsf_raw
        except Exception:
            pass
        summary['dsf'] = dsf

        # PIP stats
        pip = {}
        try:
            if self.pip_processor:
                pip['frames_processed'] = int(self.pip_processor.statistics.get('frames_processed', 0)) if getattr(self.pip_processor, 'statistics', None) else 0
                # Counts by intent
                high = len(self.pip_processor.crossing_alerts) if getattr(self.pip_processor, 'crossing_alerts', None) else 0
                pip['high_intent_count'] = high
                # For low/moderate estimation we scan last statistics if available
                hist = [0, 0, 0]
                # attempt to build histogram from last known system stats
                stats = getattr(self.pip_processor, 'statistics', {}) or {}
                intent_hist = stats.get('intent_histogram') or stats.get('intent_hist')
                if intent_hist:
                    pip['intent_hist'] = intent_hist
                    hist = intent_hist
                # Add more stats if available
                pip['avg_inference_time_ms'] = stats.get('avg_inference_time_ms', 0)
                pip['fps'] = stats.get('fps', 0)
                pip['alert_stats'] = stats.get('alert_stats', {})
                # optional timeseries of alerts (if system collected it)
                pip['alert_timeseries'] = stats.get('alert_timeseries')
                pip['low_intent_count'] = hist[0] if len(hist) >= 1 else 0
                pip['moderate_intent_count'] = hist[1] if len(hist) >= 2 else 0
                pip['high_intent_count'] = hist[2] if len(hist) >= 3 else high
                # Grab last annotated frame if available
                try:
                    annotated = self.pip_processor.get_annotated_frame()
                    pip['annotated_image'] = annotated
                except Exception:
                    pip['annotated_image'] = None
        except Exception:
            pip = {}
        summary['pip'] = pip

        # Generic metadata
        summary['duration_s'] = dsf.get('duration_seconds', 0) if isinstance(dsf, dict) else 0
        summary['frames'] = pip.get('frames_processed', 0)

        # Provide convenience top-level aliases
        summary['annotated_image'] = pip.get('annotated_image')

        return summary