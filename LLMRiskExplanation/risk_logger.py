"""
risk_logger.py
──────────────
Writes a JSON-lines log of every risk event that triggered the LLM.
Each line is a self-contained JSON object so the file is easy to
parse, grep, and feed back into the model for post-trip review.

Log location:  NeuroDrive/LLMRiskExplanation/logs/risk_log_YYYYMMDD.jsonl
"""

from __future__ import annotations
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .risk_snapshot import RiskSnapshot


class RiskLogger:
    """Append-only JSON-lines logger."""

    def __init__(self, log_dir: Optional[str] = None):
        if log_dir is None:
            # Place logs inside the LLMRiskExplanation folder
            _here    = Path(__file__).resolve().parent
            log_dir  = str(_here / "logs")

        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._file   = None
        self._date   = None
        self._ensure_file()

    # ─────────────────────────────────────────────────────────────────────────

    def log(self, snapshot: RiskSnapshot, explanation: Optional[str]) -> None:
        """Write one event record to the daily log file."""
        self._ensure_file()
        record = {
            "ts":          snapshot.timestamp,
            "datetime":    datetime.fromtimestamp(snapshot.timestamp).isoformat(
                               timespec="seconds"
                           ),
            "speed_kmh":   snapshot.ego_speed_kmh,
            "alerts": {
                "fcw_critical":       snapshot.fcw_critical,
                "fcw_depth_m":        snapshot.fcw_depth_m  if snapshot.fcw_critical else None,
                "fcw_ttc_s":          snapshot.fcw_ttc_s    if snapshot.fcw_critical else None,
                "left_bsp":           snapshot.left_bsp,
                "right_bsp":          snapshot.right_bsp,
                "bsp_distance_m":     snapshot.bsp_distance_m,
                "lane_warning":       snapshot.lane_warning,
                "lane_direction":     snapshot.lane_direction if snapshot.lane_warning else None,
                "lane_position_m":    snapshot.lane_position_m if snapshot.lane_warning else None,
                "sign_class":         snapshot.sign_class,
                "sign_confidence":    snapshot.sign_confidence if snapshot.sign_class else None,
                "active_rules_count": len(snapshot.active_rules),
                "top_rule":           snapshot.active_rules[0].get("description")
                                      if snapshot.active_rules else None,
                "driver_distracted":  snapshot.driver_distracted,
                "driver_alerts":      snapshot.driver_alerts,
                "eyes_open":          snapshot.eyes_open,
                "gaze":               snapshot.gaze,
            },
            "explanation": explanation,
        }
        try:
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()
        except Exception as exc:
            print(f"[RiskLogger] Write error: {exc}")

    def close(self) -> None:
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    # ── Private ───────────────────────────────────────────────────────────────

    def _ensure_file(self) -> None:
        today = datetime.now().strftime("%Y%m%d")
        if self._date != today or self._file is None:
            self.close()
            path = self._log_dir / f"risk_log_{today}.jsonl"
            self._file = open(path, "a", encoding="utf-8")
            self._date = today