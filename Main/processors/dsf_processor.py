"""
Driving Style Feedback processor — Module 9
Wraps DrivingStyleFeedback into the BaseProcessor pattern.

Changes from original:
  • NO HUD overlay rendered onto video — data is displayed in the bottom panel
  • Accepts rich alert state from ALL modules (FCW, BSM, lane, signs, distraction)
  • Exposes structured data dict for the UI panel to render
"""
import math
import time
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_processor import BaseProcessor
from DrivingStyleFeedback import DrivingStyleFeedback


class DrivingStyleFeedbackProcessor(BaseProcessor):
    """
    Module 9 — Driving Style Feedback.

    Runs the DSF engine in a background thread at 10 Hz.
    Does NOT render any overlay onto video frames.
    Exposes .panel_data dict for the bottom-bar DSF panel.
    """

    def __init__(self, config_path=None, speed_kmh: float = 60.0):
        # No video source needed — DSF is data-only now
        super().__init__(video_source=None)
        self.config_path       = config_path
        self.speed_kmh         = speed_kmh
        self.detection_enabled = True

        self._dsf: DrivingStyleFeedback = None
        self._lock = threading.Lock()

        # Public state for the UI panel
        self.current_score  = 100
        self.current_tier   = "Safe"
        self.event_counts   = {"harsh_accel": 0, "harsh_brake": 0, "aggressive_steer": 0}
        self.risk_weight    = 1.0
        self.distance_km    = 0.0
        self.elapsed_s      = 0.0
        self.journey_summary = None
        self.detection_status = False   # True when tier is Aggressive

        # Latest sensor values for display
        self.last_accel = 0.0
        self.last_decel = 0.0
        self.last_steer = 0.0

        # Latest alert state from other modules
        self._alert_state: dict = {}

        # Summaries output folder
        self._summaries_dir = Path("summaries")
        self._summaries_dir.mkdir(exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_alert_state(self, state: dict):
        """
        FR9.4 — forward external alerts from ALL modules to DSF engine.
        Keys accepted:
          drowsiness_alert, distraction_alert, fcw_alert,
          lane_alert, bsp_alert, sign_alert, priority_alert, pip_alert
        """
        with self._lock:
            self._alert_state.update(state)
        if self._dsf:
            # DSF engine only uses drowsiness_alert + distraction_alert for weight
            self._dsf.update_alert_state({
                "drowsiness_alert":  bool(state.get("drowsiness_alert") or
                                          state.get("distraction_alert")),
                "distraction_alert": bool(state.get("distraction_alert") or
                                          state.get("fcw_alert") or
                                          state.get("lane_alert")),
            })

    @property
    def panel_data(self) -> dict:
        """Structured data dict for the bottom-bar DSF panel."""
        with self._lock:
            alerts = dict(self._alert_state)
        return {
            "score":        self.current_score,
            "tier":         self.current_tier,
            "event_counts": dict(self.event_counts),
            "risk_weight":  self.risk_weight,
            "distance_km":  round(self.distance_km, 3),
            "elapsed_s":    self.elapsed_s,
            "accel":        self.last_accel,
            "decel":        self.last_decel,
            "steer":        self.last_steer,
            "active_alerts": alerts,
        }

    def get_summary(self) -> dict:
        return self.journey_summary or {}

    # ── BaseProcessor overrides ───────────────────────────────────────────────

    def start(self):
        """Start the DSF engine thread (no video capture)."""
        if not self.is_running:
            self.is_running = True
            import threading as _t
            self.thread = _t.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self):
        try:
            self._dsf = DrivingStyleFeedback(config_path=self.config_path)
            self._dsf.start_journey()

            sample_interval = 1.0 / 10.0   # FR9.1 — 10 Hz
            last_sample_t   = time.time()
            frame_idx       = 0

            print("[DSF] Journey started — data-only mode (10 Hz)")

            while self.is_running:
                now = time.time()

                if self.detection_enabled and (now - last_sample_t) >= sample_interval:
                    t = frame_idx / 10.0

                    # Synthetic sensor data (replace with OBD when available)
                    accel = 0.5 * math.sin(t * 0.3) + 0.2 * math.sin(t * 1.1)
                    decel = -0.3 * abs(math.sin(t * 0.5)) - 0.1
                    steer = 8.0 * math.sin(t * 0.7)

                    # Inject occasional harsh events for demo realism
                    if frame_idx % 80  == 0: accel =  3.2
                    if frame_idx % 120 == 0: decel = -3.8
                    if frame_idx % 150 == 0: steer = 22.0

                    snap = self._dsf.process_sample(
                        accel, decel, steer,
                        timestamp=now,
                        speed_ms=self.speed_kmh / 3.6,
                    )

                    # Update public state
                    self.current_score  = snap.get("score",        self.current_score)
                    self.current_tier   = snap.get("tier",         self.current_tier)
                    self.event_counts   = snap.get("event_counts", self.event_counts)
                    self.risk_weight    = snap.get("risk_weight",  1.0)
                    self.distance_km    = snap.get("distance_km",  self.distance_km)
                    self.last_accel     = accel
                    self.last_decel     = decel
                    self.last_steer     = steer
                    self.detection_status = (self.current_tier == "Aggressive")

                    if self._dsf._active:
                        self.elapsed_s = now - self._dsf._jstart

                    last_sample_t = now
                    frame_idx    += 1

                time.sleep(0.05)   # ~20 Hz loop, samples at 10 Hz

        except Exception as e:
            print(f"[DSF] Error: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=3)
        if self._dsf:
            try:
                self.journey_summary = self._dsf.stop_journey()
                self._save_summary_txt(self.journey_summary)
            except Exception:
                pass

    def _save_summary_txt(self, summary: dict):
        """Save journey summary as a rich human-readable .txt file to summaries/."""
        try:
            self._summaries_dir.mkdir(parents=True, exist_ok=True)
            existing = list(self._summaries_dir.glob("summary_*.txt"))
            nums = []
            for p in existing:
                try:
                    nums.append(int(p.stem.split("_")[1]))
                except (IndexError, ValueError):
                    pass
            next_num = max(nums) + 1 if nums else 1
            out_path = self._summaries_dir / f"summary_{next_num}.txt"

            with self._lock:
                alert_state = dict(self._alert_state)

            lines = self._build_txt(summary, alert_state, next_num)
            with open(out_path, "w") as fh:
                fh.write("\n".join(lines))
            print(f"[DSF] Summary saved → {out_path}")
        except Exception as e:
            print(f"[DSF] Failed to save summary: {e}")

    def _build_txt(self, s: dict, alerts: dict, run_num: int) -> list:
        """Build the full text summary lines."""
        from datetime import datetime

        tier  = s.get("tier", "Safe")
        score = s.get("final_score", 0)
        ec    = s.get("event_counts", {})
        dur   = s.get("duration_minutes", 0)
        dist  = s.get("distance_km", 0)
        ra    = s.get("rolling_avg_score")
        trend = s.get("trend")

        harsh_accel = ec.get("harsh_accel", 0)
        harsh_brake = ec.get("harsh_brake", 0)
        aggr_steer  = ec.get("aggressive_steer", 0)
        total_harsh = harsh_accel + harsh_brake + aggr_steer

        # Module alert counts from the session
        fcw_triggered      = alerts.get("fcw_alert", False)
        lane_triggered     = alerts.get("lane_alert", False)
        bsp_triggered      = alerts.get("bsp_alert", False)
        sign_triggered     = alerts.get("sign_alert", False)
        priority_triggered = alerts.get("priority_alert", False)
        driver_triggered   = alerts.get("distraction_alert", False)
        pip_triggered      = alerts.get("pip_alert", False)

        tier_emoji = {"Safe": "✅", "Moderate": "⚡", "Aggressive": "🚨"}.get(tier, "")
        trend_str  = ("↑ Improving" if trend == "improving" else
                      "↓ Regressing" if trend == "regressing" else "→ Stable")

        L = []
        W = 66

        def sep(char="="): L.append(char * W)
        def blank():       L.append("")
        def title(t):
            pad = (W - len(t) - 2) // 2
            L.append("=" * W)
            L.append("|" + " " * pad + t + " " * (W - pad - len(t) - 2) + "|")
        def row(label, value, width=W):
            line = f"  {label:<28}{value}"
            L.append(line[:width])

        sep()
        title(f"NeuroDrive — Journey Summary #{run_num}")
        title("Module 9: Driving Style Feedback  (FR9.5)")
        sep()
        blank()

        # ── Session Info ──────────────────────────────────────────────────────
        L.append("SESSION INFORMATION")
        sep("-")
        row("Run Number:",          f"#{run_num}")
        row("Journey Start:",       s.get("journey_start", "N/A"))
        row("Journey End:",         s.get("journey_end",   "N/A"))
        row("Duration:",            f"{dur:.1f} minutes")
        row("Distance Covered:",    f"{dist:.3f} km")
        if ra:
            row("7-Trip Rolling Avg:", f"{ra:.1f} / 100")
        blank()

        # ── Overall Score ─────────────────────────────────────────────────────
        L.append("OVERALL DRIVING SCORE")
        sep("-")
        row("Final Score:",         f"{score} / 100  {tier_emoji} {tier}")
        row("Trend:",               trend_str)
        row("Risk Weight Applied:", f"{self.risk_weight:.2f}x")
        blank()

        # ── Driving Behaviour ─────────────────────────────────────────────────
        L.append("DRIVING BEHAVIOUR EVENTS")
        sep("-")
        row("Harsh Accelerations:", str(harsh_accel))
        row("Harsh Brakings:",      str(harsh_brake))
        row("Aggressive Steers:",   str(aggr_steer))
        row("Total Harsh Events:",  str(total_harsh))
        blank()

        # Top events
        top = s.get("top_events", [])
        if top:
            L.append("  Top 5 Worst Events:")
            for i, ev in enumerate(top[:5], 1):
                ts  = ev.get("timestamp", "")[:19].replace("T", " ")
                typ = ev.get("type", "").replace("_", " ").title()
                mag = ev.get("magnitude", 0)
                unit = "m/s²" if "steer" not in ev.get("type","") else "deg/s"
                L.append(f"    {i}. {typ:<22} {mag:+.2f} {unit}  @ {ts}")
        blank()

        # ── Module Alerts ─────────────────────────────────────────────────────
        L.append("MODULE ALERT SUMMARY")
        sep("-")
        def yn(v): return "YES ⚠" if v else "No  ✓"
        row("Forward Collision Warning:",   yn(fcw_triggered))
        row("Lane Departure Warning:",      yn(lane_triggered))
        row("Blind Spot Monitoring:",       yn(bsp_triggered))
        row("Traffic Sign Detection:",      yn(sign_triggered))
        row("Priority Rules Alert:",        yn(priority_triggered))
        row("Driver Distraction:",          yn(driver_triggered))
        row("Pedestrian Intent Alert:",     yn(pip_triggered))
        blank()

        # ── Dynamic Recommendations ───────────────────────────────────────────
        L.append("PERSONALISED RECOMMENDATIONS")
        sep("-")
        recs = self._dynamic_recommendations(
            harsh_accel, harsh_brake, aggr_steer, total_harsh, score, tier,
            fcw_triggered, lane_triggered, bsp_triggered, driver_triggered,
            pip_triggered, priority_triggered, sign_triggered, dur, dist
        )
        for i, rec in enumerate(recs, 1):
            # Word-wrap at 60 chars
            words = rec.split()
            line, wrapped = f"  {i}. ", []
            for w in words:
                if len(line) + len(w) + 1 > 64:
                    wrapped.append(line)
                    line = "     " + w + " "
                else:
                    line += w + " "
            if line.strip():
                wrapped.append(line)
            L.extend(wrapped)
            blank()

        # ── Score Trend ───────────────────────────────────────────────────────
        series = s.get("score_series", [])
        if series:
            L.append("SCORE TREND (sampled)")
            sep("-")
            # Show 10 evenly-spaced samples
            n = len(series)
            step = max(1, n // 10)
            samples = series[::step][:10]
            bar_line = "  "
            for v in samples:
                filled = int((v / 100) * 10)
                bar_line += f"[{'█' * filled}{'░' * (10 - filled)}] {v:.0f}  "
            L.append(bar_line[:W])
        blank()

        sep("═")
        L.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        L.append(f"  NeuroDrive ADAS — Module 9 Driving Style Feedback")
        sep("═")

        return L

    def _dynamic_recommendations(
        self, harsh_accel, harsh_brake, aggr_steer, total_harsh,
        score, tier, fcw, lane, bsp, driver, pip, priority, sign,
        dur, dist
    ) -> list:
        """Generate fully dynamic, session-specific recommendations."""
        recs = []

        # Score-based opening
        if score >= 90:
            recs.append(
                f"Excellent session! Score of {score}/100 reflects consistently "
                f"smooth driving. Keep maintaining this standard."
            )
        elif score >= 70:
            recs.append(
                f"Good session with a score of {score}/100. A few rough moments "
                f"held you back — focus on the specific areas below."
            )
        else:
            recs.append(
                f"Score of {score}/100 indicates significant room for improvement. "
                f"Review each alert category below carefully."
            )

        # Harsh acceleration
        if harsh_accel > 0:
            if harsh_accel >= 5:
                recs.append(
                    f"You triggered {harsh_accel} harsh accelerations this session. "
                    f"Ease into the throttle gradually — rapid acceleration wastes "
                    f"fuel and increases wear on the drivetrain."
                )
            else:
                recs.append(
                    f"{harsh_accel} harsh acceleration event(s) detected. "
                    f"Try to anticipate gaps in traffic earlier so you can "
                    f"accelerate smoothly rather than in bursts."
                )

        # Harsh braking
        if harsh_brake > 0:
            if harsh_brake >= 5:
                recs.append(
                    f"{harsh_brake} harsh braking events recorded — this is the "
                    f"highest-risk behaviour in your session. Increase your following "
                    f"distance so you can brake gently and progressively."
                )
            else:
                recs.append(
                    f"{harsh_brake} harsh braking event(s) detected. "
                    f"Scan further ahead and anticipate stops earlier to avoid "
                    f"last-second hard braking."
                )

        # Aggressive steering
        if aggr_steer > 0:
            recs.append(
                f"{aggr_steer} aggressive steering input(s) detected. "
                f"Reduce speed before corners and apply smooth, progressive "
                f"steering inputs — this improves tyre life and passenger comfort."
            )

        # FCW
        if fcw:
            recs.append(
                "Forward Collision Warning was triggered during this session. "
                "Maintain at least a 3-second following gap and reduce speed "
                "when approaching slow-moving or stationary traffic."
            )

        # Lane departure
        if lane:
            recs.append(
                "Lane Departure Warning activated. Check for fatigue or distraction "
                "— if you feel tired, take a break. Always signal before changing lanes."
            )

        # Blind spot
        if bsp:
            recs.append(
                "Blind Spot alerts were triggered. Always check mirrors AND "
                "physically glance over your shoulder before lane changes, "
                "especially on multi-lane roads."
            )

        # Driver distraction
        if driver:
            recs.append(
                "Driver distraction was detected during this session. "
                "Keep your eyes on the road at all times — even a 2-second "
                "glance away at 60 km/h covers 33 metres blind."
            )

        # Pedestrian
        if pip:
            recs.append(
                "Pedestrian crossing intent was detected. Always slow down "
                "near crossings and be prepared to stop — pedestrians have "
                "right of way at marked crossings."
            )

        # Priority rules
        if priority:
            recs.append(
                "Priority rule alerts were active (emergency vehicles or "
                "junction rules). Always yield to emergency vehicles immediately "
                "and follow right-of-way rules at intersections."
            )

        # Traffic signs
        if sign:
            recs.append(
                "Traffic signs were detected during the session. Ensure you "
                "are actively reading and complying with all posted signs, "
                "especially speed limits and stop signs."
            )

        # Short trip with no issues
        if total_harsh == 0 and not any([fcw, lane, bsp, driver, pip]):
            recs.append(
                f"Clean session — zero harsh events and no safety alerts over "
                f"{dur:.1f} minutes and {dist:.2f} km. Outstanding driving!"
            )

        # Distance-based tip
        if dist > 5:
            recs.append(
                f"Over {dist:.1f} km driven — on longer journeys, take a "
                f"15-minute break every 2 hours to maintain concentration."
            )

        return recs
