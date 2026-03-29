"""
overtake_logic.py
─────────────────
Core overtake-safety analysis engine.

Performs three sequential checks:
  1. Blind-spot clear on the chosen side (using BlindSpotProcessor state)
  2. Front-side clear  — vehicles in the outer 30-100 % of the front frame
                        on the chosen side, with distance estimate
  3. Approaching from behind — vehicles closing in the side-mirror feed

All results are packed into OvertakeResult and exposed for the UI and
the LLM Risk Explainer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np


# ── Enums ──────────────────────────────────────────────────────────────────────

class OvertakeState(Enum):
    IDLE        = "idle"          # not requested
    WAITING     = "waiting"       # button pressed, awaiting side selection
    CHECKING    = "checking"      # actively running checks
    SAFE        = "safe"          # all checks passed
    CAUTION     = "caution"       # marginal — proceed carefully
    UNSAFE      = "unsafe"        # do NOT overtake
    EXPIRED     = "expired"       # result is older than TTL


class OvertakeSide(Enum):
    LEFT  = "left"
    RIGHT = "right"


# ── Result dataclass ────────────────────────────────────────────────────────────

@dataclass
class OvertakeResult:
    state:               OvertakeState     = OvertakeState.IDLE
    side:                Optional[OvertakeSide] = None
    timestamp:           float             = 0.0

    # Check 1 — blind spot
    blindspot_clear:     bool              = True
    blindspot_vehicles:  int               = 0
    blindspot_distance_m: Optional[float] = None

    # Check 2 — front-side zone
    front_clear:         bool              = True
    front_vehicle_dist_m: Optional[float] = None
    front_vehicle_count: int               = 0
    safe_gap_seconds:    Optional[float]  = None   # estimated time gap

    # Check 3 — approaching from behind
    approaching_from_behind: bool         = False
    approach_speed_mps:      float        = 0.0

    # Composite
    can_overtake:        bool              = False
    reason:              str               = ""

    # Raw data for LLM prompt
    ego_speed_kmh:       float             = 0.0
    notes:               list = field(default_factory=list)


# ── Safety thresholds ──────────────────────────────────────────────────────────

# Minimum gap (seconds) between your car and the front vehicle at current speed
MIN_GAP_SECONDS     = 4.0   # need at least 4 s of clear road ahead to overtake
# Front-zone: fraction of frame width that counts as "the chosen side"
FRONT_ZONE_FRACTION = 0.30  # right 30 % → right side, left 30 % → left side
# YOLO vehicle classes (COCO)
VEHICLE_CLASSES     = {2, 3, 5, 7}   # car, motorcycle, bus, truck
# Focal length proxy for depth from bbox (pixels)
FOCAL_LENGTH        = 700.0
REAL_CAR_HEIGHT_M   = 1.5


# ── Depth helper ───────────────────────────────────────────────────────────────

def _depth_from_bbox(bbox, frame_shape) -> float:
    """Quick pinhole-model depth estimate (metres) from bounding-box height."""
    x1, y1, x2, y2 = bbox
    h = max(1.0, y2 - y1)
    img_h = frame_shape[0]
    depth = (FOCAL_LENGTH * REAL_CAR_HEIGHT_M) / h
    # perspective correction by vertical position
    vp = y2 / img_h
    if vp > 0.85:
        depth *= 0.70
    elif vp > 0.75:
        depth *= 0.85
    elif vp < 0.65:
        depth *= 1.20
    return float(np.clip(depth, 1.0, 120.0))


# ── Main analyser ───────────────────────────────────────────────────────────────

class OvertakeAnalyser:
    """
    Stateless analysis object — call analyse() every time you want a fresh
    OvertakeResult.  All processor state is read from the live processor
    objects passed in (no internal caching).
    """

    def __init__(self, yolo=None):
        """
        Parameters
        ──────────
        yolo : ultralytics YOLO instance (optional).  If provided the
               analyser can scan the live front frame for vehicles.
               If None, it falls back to using the FCW processor's
               detection state.
        """
        self._yolo = yolo

    # ── Public entry point ────────────────────────────────────────────────────

    def analyse(
        self,
        side:                OvertakeSide,
        left_bsp_proc,
        right_bsp_proc,
        fcw_proc,
        front_frame:         Optional[np.ndarray],
        ego_speed_kmh:       float = 0.0,
    ) -> OvertakeResult:
        """
        Run all three checks and return a complete OvertakeResult.
        """
        result = OvertakeResult(
            state     = OvertakeState.CHECKING,
            side      = side,
            timestamp = time.time(),
            ego_speed_kmh = ego_speed_kmh,
        )

        # ── Check 1: Blind spot ───────────────────────────────────────────────
        self._check_blindspot(result, side, left_bsp_proc, right_bsp_proc)

        # ── Check 2: Front-side zone ──────────────────────────────────────────
        self._check_front(result, side, front_frame, ego_speed_kmh, fcw_proc)

        # ── Check 3: Approaching from behind ─────────────────────────────────
        self._check_approaching(result, side, left_bsp_proc, right_bsp_proc)

        # ── Final verdict ─────────────────────────────────────────────────────
        self._compute_verdict(result)

        return result

    # ── Check 1 ───────────────────────────────────────────────────────────────

    def _check_blindspot(self, result, side, left_proc, right_proc):
        proc = left_proc if side == OvertakeSide.LEFT else right_proc
        if proc is None or not proc.is_running:
            result.notes.append("Blind-spot camera unavailable — check skipped.")
            return

        result.blindspot_vehicles = proc.vehicle_count or 0
        result.blindspot_distance_m = proc.vehicle_distance
        result.blindspot_clear = result.blindspot_vehicles == 0

        if not result.blindspot_clear:
            dist_txt = (f"{result.blindspot_distance_m:.1f} m away"
                        if result.blindspot_distance_m else "distance unknown")
            result.notes.append(
                f"Blind spot ({side.value}): {result.blindspot_vehicles} "
                f"vehicle(s) detected, {dist_txt}."
            )
        else:
            result.notes.append(f"Blind spot ({side.value}): clear.")

    # ── Check 2 ───────────────────────────────────────────────────────────────

    def _check_front(self, result, side, front_frame, ego_speed_kmh, fcw_proc):
        if front_frame is None:
            # Fall back to FCW processor state
            if fcw_proc and hasattr(fcw_proc, 'critical') and fcw_proc.critical:
                result.front_clear = False
                result.front_vehicle_count = 1
                result.notes.append("Front camera unavailable; FCW reports critical alert.")
            else:
                result.notes.append("Front camera unavailable — front check skipped.")
            return

        h, w = front_frame.shape[:2]

        # Define the side zone: outer 30-100 % of frame on chosen side
        if side == OvertakeSide.RIGHT:
            zone_x1 = int(w * (1.0 - FRONT_ZONE_FRACTION))
            zone_x2 = w
        else:
            zone_x1 = 0
            zone_x2 = int(w * FRONT_ZONE_FRACTION)

        # Only look in the lower 60 % of the frame (road level)
        zone_y1 = int(h * 0.40)
        zone_y2 = h

        vehicles_in_zone = []

        if self._yolo is not None:
            try:
                res = self._yolo(
                    front_frame, imgsz=640, conf=0.30, verbose=False
                )
                if res and len(res[0].boxes):
                    for box in res[0].boxes:
                        cls = int(box.cls[0])
                        if cls not in VEHICLE_CLASSES:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        # Must be in the side zone and at road level
                        if (zone_x1 <= cx <= zone_x2 and
                                zone_y1 <= cy <= zone_y2):
                            depth = _depth_from_bbox((x1, y1, x2, y2),
                                                     front_frame.shape)
                            vehicles_in_zone.append({
                                'bbox':  (x1, y1, x2, y2),
                                'depth': depth,
                            })
            except Exception as e:
                result.notes.append(f"Front YOLO scan error: {e}")

        result.front_vehicle_count = len(vehicles_in_zone)
        result.front_clear = result.front_vehicle_count == 0

        if vehicles_in_zone:
            # Closest vehicle
            closest = min(vehicles_in_zone, key=lambda v: v['depth'])
            result.front_vehicle_dist_m = closest['depth']
            ego_mps = max(ego_speed_kmh / 3.6, 1.0)
            result.safe_gap_seconds = result.front_vehicle_dist_m / ego_mps
            result.notes.append(
                f"Front-side zone ({side.value}): {result.front_vehicle_count} "
                f"vehicle(s), closest at {result.front_vehicle_dist_m:.1f} m "
                f"(~{result.safe_gap_seconds:.1f} s gap at current speed)."
            )
        else:
            result.notes.append(f"Front-side zone ({side.value}): clear.")

    # ── Check 3 ───────────────────────────────────────────────────────────────

    def _check_approaching(self, result, side, left_proc, right_proc):
        """
        Re-uses the blind-spot feed.  If vehicle_count > 0 AND the processor
        stores a vehicle distance that is shrinking (approaching), flag it.
        For now we use a conservative heuristic: any vehicle within 8 m in
        the blind-spot camera is treated as 'approaching'.
        """
        proc = left_proc if side == OvertakeSide.LEFT else right_proc
        if proc is None or not proc.is_running:
            return

        if proc.vehicle_count and proc.vehicle_count > 0:
            dist = proc.vehicle_distance
            if dist is not None and dist < 8.0:
                result.approaching_from_behind = True
                result.notes.append(
                    f"Vehicle approaching from behind on {side.value} "
                    f"side at ~{dist:.1f} m."
                )

    # ── Verdict ───────────────────────────────────────────────────────────────

    def _compute_verdict(self, result: OvertakeResult):
        unsafe_reasons  = []
        caution_reasons = []

        # Hard blocks
        if not result.blindspot_clear:
            unsafe_reasons.append("vehicle in blind spot")
        if result.approaching_from_behind:
            unsafe_reasons.append("vehicle approaching from behind")

        # Distance-based
        if not result.front_clear:
            if result.front_vehicle_dist_m is not None:
                if result.front_vehicle_dist_m < 15.0:
                    unsafe_reasons.append(
                        f"vehicle only {result.front_vehicle_dist_m:.0f} m ahead "
                        "in target lane — too close"
                    )
                elif result.front_vehicle_dist_m < 30.0:
                    caution_reasons.append(
                        f"vehicle {result.front_vehicle_dist_m:.0f} m ahead "
                        "in target lane — marginal gap"
                    )
            else:
                caution_reasons.append("vehicle detected ahead in target lane")

        # Gap-time check
        if result.safe_gap_seconds is not None:
            if result.safe_gap_seconds < 2.0:
                unsafe_reasons.append(
                    f"gap only {result.safe_gap_seconds:.1f} s "
                    f"(need ≥ {MIN_GAP_SECONDS} s)"
                )
            elif result.safe_gap_seconds < MIN_GAP_SECONDS:
                caution_reasons.append(
                    f"gap is {result.safe_gap_seconds:.1f} s "
                    f"(recommended ≥ {MIN_GAP_SECONDS} s)"
                )

        if unsafe_reasons:
            result.state       = OvertakeState.UNSAFE
            result.can_overtake = False
            result.reason      = "UNSAFE — " + "; ".join(unsafe_reasons) + "."
        elif caution_reasons:
            result.state       = OvertakeState.CAUTION
            result.can_overtake = False
            result.reason      = "CAUTION — " + "; ".join(caution_reasons) + "."
        else:
            result.state       = OvertakeState.SAFE
            result.can_overtake = True
            result.reason      = (
                f"Path appears clear on the {result.side.value}. "
                "Proceed with caution and signal before manoeuvring."
            )