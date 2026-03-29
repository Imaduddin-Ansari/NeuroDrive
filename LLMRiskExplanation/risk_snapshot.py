"""
risk_snapshot.py
────────────────
Dataclass that collects one moment of state from every NeuroDrive
processor and decides whether that state is interesting enough to
send to the LLM.

Three trigger rules (any one fires the LLM):
  1. Two or more modules are alerting at the same time.
  2. Driver is distracted AND at least one hazard exists on the road.
  3. A critical-priority traffic rule is active AND speed > 30 km/h.
  4. (NEW) An overtake assessment has just completed (any verdict).

The snapshot also builds the prompt that is sent to Llama.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskSnapshot:
    """Complete state snapshot passed to the LLM."""

    timestamp: float = 0.0

    # ── Forward Collision Warning ─────────────────────────────────────────────
    fcw_critical: bool       = False
    fcw_depth_m:  float      = 0.0
    fcw_ttc_s:    float      = 999.0

    # ── Blind Spot Monitoring ─────────────────────────────────────────────────
    left_bsp:       bool           = False
    right_bsp:      bool           = False
    bsp_distance_m: Optional[float] = None

    # ── Lane Departure Warning ────────────────────────────────────────────────
    lane_warning:    bool  = False
    lane_direction:  str   = "F"
    lane_position_m: float = 0.0

    # ── Traffic Sign Detection ────────────────────────────────────────────────
    sign_class:      Optional[str] = None
    sign_confidence: float         = 0.0

    # ── Priority-Based Rules ──────────────────────────────────────────────────
    active_rules: list = field(default_factory=list)

    # ── Driver Distraction ────────────────────────────────────────────────────
    driver_distracted: bool  = False
    driver_alerts:     list  = field(default_factory=list)
    eyes_open:         bool  = True
    gaze:              str   = "straight"
    yaw_status:        str   = "straight"

    # ── Vehicle context ───────────────────────────────────────────────────────
    ego_speed_kmh: float = 0.0

    # ── Overtake Assistance (NEW) ─────────────────────────────────────────────
    overtake_requested:         bool            = False
    overtake_side:              Optional[str]   = None   # 'left' | 'right'
    overtake_verdict:           Optional[str]   = None   # 'safe' | 'caution' | 'unsafe'
    overtake_reason:            Optional[str]   = None
    overtake_blindspot_clear:   bool            = True
    overtake_front_clear:       bool            = True
    overtake_front_dist_m:      Optional[float] = None
    overtake_gap_seconds:       Optional[float] = None
    overtake_approaching_rear:  bool            = False
    overtake_notes:             list = field(default_factory=list)

    # ─────────────────────────────────────────────────────────────────────────

    def _road_hazard_count(self) -> int:
        """Number of active road-level hazards (excludes driver state)."""
        n = 0
        if self.fcw_critical:               n += 1
        if self.left_bsp or self.right_bsp: n += 1
        if self.lane_warning:               n += 1
        if self.active_rules:               n += 1
        if self.sign_class and self.sign_confidence >= 0.85:
            n += 1
        return n

    def active_alert_count(self) -> int:
        """Total active alerts including driver state."""
        n = self._road_hazard_count()
        if self.driver_distracted:
            n += 1
        return n

    def should_trigger(self) -> bool:
        """
        Return True when the snapshot is worth sending to the LLM.

        Rule 1 — multi-alert: 3+ distinct modules alerting simultaneously.
                  This is intentionally strict — 2 alerts are handled by
                  the consecutive-gate in main_window before submit() is
                  even called, so this guards against snapshot-level spurious
                  triggers (e.g. a single module that counts as 2 alerts).
        Rule 2 — distracted + TWO hazards: driver is not watching and there
                  are at least two independent road hazards present, making
                  the combined risk genuinely high.
        Rule 3 — critical rule + speed: a critical-priority traffic rule fires
                  while the vehicle is moving faster than 30 km/h.
        Rule 4 — overtake verdict just arrived (any verdict).
        """
        # Rule 1 — require 3 distinct alert sources
        if self.active_alert_count() >= 3:
            return True

        # Rule 2 — distracted driver + at least 2 independent road hazards
        if self.driver_distracted and self._road_hazard_count() >= 2:
            return True

        # Rule 3
        critical_rules = [
            r for r in self.active_rules
            if r.get("priority") == "critical"
        ]
        if critical_rules and self.ego_speed_kmh > 30:
            return True

        # Rule 4 — overtake assessment
        if self.overtake_requested and self.overtake_verdict is not None:
            return True

        return False

    def to_prompt(self) -> str:
        """
        Build the Llama prompt for this snapshot.

        The system instruction is short so the model stays fast on a
        local 3 B parameter deployment.  We ask for one or two sentences,
        action-oriented, no sensor labels.
        """
        lines: list[str] = []
        lines.append(
            "You are a driving co-pilot. Describe the current risk in "
            "exactly 1-2 sentences and tell the driver exactly what to do. "
            "Be specific and direct. Never use more than 2 sentences. "
            "Do not restate sensor names or repeat facts "
            "already obvious to a driver. Do not use bullet points.\n"
        )
        lines.append("Current driving situation:")

        # FCW
        if self.fcw_critical:
            lines.append(
                f"- Collision imminent: vehicle {self.fcw_depth_m:.1f} m ahead, "
                f"time-to-collision {self.fcw_ttc_s:.1f} s"
            )

        # Blind spots
        if self.left_bsp and self.right_bsp:
            lines.append("- Vehicles detected in BOTH blind spots")
        elif self.left_bsp:
            dist = (f", approximately {self.bsp_distance_m:.0f} m away"
                    if self.bsp_distance_m else "")
            lines.append(f"- Vehicle in LEFT blind spot{dist}")
        elif self.right_bsp:
            dist = (f", approximately {self.bsp_distance_m:.0f} m away"
                    if self.bsp_distance_m else "")
            lines.append(f"- Vehicle in RIGHT blind spot{dist}")

        # Lane departure
        if self.lane_warning:
            direction = "left" if self.lane_direction == "L" else "right"
            lines.append(
                f"- Drifting {direction}, "
                f"{abs(self.lane_position_m):.1f} m off lane centre"
            )

        # Traffic sign
        if self.sign_class and self.sign_confidence >= 0.70:
            lines.append(f"- Traffic sign detected: {self.sign_class}")

        # Priority rules — show only the highest-priority one
        if self.active_rules:
            top_rule = self.active_rules[0]
            desc     = top_rule.get("description", "")
            priority = top_rule.get("priority", "")
            if desc:
                lines.append(f"- Traffic rule ({priority}): {desc}")

        # Driver state
        if self.driver_distracted:
            if not self.eyes_open:
                lines.append("- Driver eyes are CLOSED")
            elif self.gaze != "straight":
                lines.append(f"- Driver gaze: looking {self.gaze}, not at road")
            elif self.yaw_status not in ("straight",):
                lines.append(f"- Driver head turned {self.yaw_status.replace('_', ' ')}")
            else:
                lines.append("- Driver is distracted")

        # ── Overtake Assessment (NEW) ─────────────────────────────────────────
        if self.overtake_requested and self.overtake_verdict is not None:
            side_txt = self.overtake_side or "unknown"
            lines.append(
                f"- Driver requested overtake to the {side_txt}. "
                f"Assessment: {self.overtake_verdict.upper()}"
            )
            if not self.overtake_blindspot_clear:
                lines.append(
                    f"  • Blind spot ({side_txt}) is OCCUPIED"
                )
            else:
                lines.append(f"  • Blind spot ({side_txt}): clear")

            if not self.overtake_front_clear:
                dist_txt = (f"{self.overtake_front_dist_m:.0f} m"
                            if self.overtake_front_dist_m else "unknown distance")
                gap_txt  = (f", {self.overtake_gap_seconds:.1f} s gap"
                            if self.overtake_gap_seconds else "")
                lines.append(
                    f"  • Vehicle in target lane {dist_txt} ahead{gap_txt}"
                )
            else:
                lines.append("  • Target lane ahead: clear")

            if self.overtake_approaching_rear:
                lines.append("  • Vehicle approaching rapidly from behind")

            if self.overtake_reason:
                lines.append(f"  • Summary: {self.overtake_reason}")

        # Speed context — only include if it adds meaning
        if self.ego_speed_kmh > 0:
            lines.append(f"- Vehicle speed: {self.ego_speed_kmh:.0f} km/h")

        lines.append("\nResponse (max 2 sentences, action-oriented, no lists):")
        return "\n".join(lines)