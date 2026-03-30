#!/usr/bin/env python3
"""
driving_style_feedback.py
=========================
NeuroDrive ADAS — Module 9: Driving Style Feedback
===================================================

FR9.1  Analyze acceleration (>2.5 m/s²), braking (<-3.0 m/s²), steering
       (>15 deg/s) in real time at 10 Hz.
FR9.2  Classify driving pattern: Aggressive (0–40), Moderate (41–70),
       Safe (71–100) based on 95th-percentile statistical thresholds.
FR9.3  Periodic feedback every 15 minutes of continuous driving.
FR9.4  Integrate with FCW / Drowsiness modules; increase risk weight by
       25 % when distraction or drowsiness is detected.
FR9.5  End-of-journey summary: distance, duration, score, harsh-event
       counts, improvement recommendations.

Public API
----------
    dsf = DrivingStyleFeedback(config_path="dsf_config.yaml")
    dsf.start_journey()
    snapshot = dsf.process_sample(accel, decel, steer_rate,
                                   speed_ms=v, timestamp=t)
    dsf.update_alert_state({"drowsiness_alert": True})
    frame   = dsf.render_overlay(frame)
    summary = dsf.stop_journey()

CLI
---
    python driving_style_feedback.py --video test.mp4 --speed 60
    python driving_style_feedback.py --synthetic --duration 120

Author : NeuroDrive Team
Date   : 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import yaml
    _YAML = True
except ImportError:
    _YAML = False

logging.basicConfig(level=logging.INFO, format="[DSF %(levelname)s] %(message)s")
log = logging.getLogger("dsf")

# ──────────────────────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG: Dict = {
    "sensor": {
        "sample_rate_hz":   10,
        "gap_threshold_s":  0.5,
        "max_accel_ms2":    9.8,
        "max_steer_degs":   720.0,
        "calibration": {
            "accel_scale": 1.0,
            "decel_scale": 1.0,
            "steer_scale": 1.0,
        },
    },
    "thresholds": {
        "harsh_accel_ms2":  2.5,
        "harsh_brake_ms2": -3.0,
        "harsh_steer_degs": 15.0,
        "dedup_window_s":   1.0,
    },
    "scoring": {
        "window_s":             10,
        "ewma_decay":           0.95,
        "tier_aggressive_max":  40,
        "tier_moderate_max":    70,
        "penalty_accel":        10,
        "penalty_brake":        12,
        "penalty_steer":        8,
    },
    "risk": {
        "impairment_weight": 1.25,
    },
    "feedback": {
        "periodic_interval_s": 900,
        "output_dir":          "dsf_output",
        "history_file":        "journey_history.jsonl",
        "rolling_window":      7,
        "trend_threshold":     10,
    },
    "recommendations": {
        "harsh_accel":      "Ease off the accelerator — smooth starts save fuel and reduce wear.",
        "harsh_brake":      "Increase following distance to allow earlier, gentler braking.",
        "aggressive_steer": "Slow before corners; steer smoothly through the turn.",
        "default":          "Maintain a steady speed and anticipate the traffic ahead.",
    },
}

_CFG_RANGES: Dict[str, Tuple] = {
    "sensor.sample_rate_hz":            (1,      100),
    "sensor.gap_threshold_s":           (0.1,    5.0),
    "sensor.max_accel_ms2":             (5.0,    20.0),
    "sensor.max_steer_degs":            (180.0,  1440.0),
    "sensor.calibration.accel_scale":   (0.1,    10.0),
    "sensor.calibration.decel_scale":   (0.1,    10.0),
    "sensor.calibration.steer_scale":   (0.1,    10.0),
    "thresholds.harsh_accel_ms2":       (0.5,    9.8),
    "thresholds.harsh_brake_ms2":       (-9.8,  -0.5),
    "thresholds.harsh_steer_degs":      (5.0,    180.0),
    "thresholds.dedup_window_s":        (0.1,    5.0),
    "scoring.window_s":                 (5,      60),
    "scoring.ewma_decay":               (0.5,    0.999),
    "scoring.tier_aggressive_max":      (10,     49),
    "scoring.tier_moderate_max":        (50,     90),
    "scoring.penalty_accel":            (1,      50),
    "scoring.penalty_brake":            (1,      50),
    "scoring.penalty_steer":            (1,      50),
    "risk.impairment_weight":           (1.0,    3.0),
    "feedback.periodic_interval_s":     (60,     99999),
    "feedback.rolling_window":          (1,      100),
    "feedback.trend_threshold":         (1,      50),
}

# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

class SensorSample:
    __slots__ = ("timestamp", "accel", "decel", "steer_rate", "is_gap_fill")
    def __init__(self, timestamp, accel, decel, steer_rate, is_gap_fill=False):
        self.timestamp   = timestamp
        self.accel       = accel
        self.decel       = decel
        self.steer_rate  = steer_rate
        self.is_gap_fill = is_gap_fill


class HarshEvent:
    __slots__ = ("event_type", "timestamp", "magnitude", "window_id")
    def __init__(self, event_type, timestamp, magnitude, window_id):
        self.event_type = event_type
        self.timestamp  = timestamp
        self.magnitude  = magnitude
        self.window_id  = window_id


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def classify_tier(score: int, cfg: Optional[Dict] = None) -> str:
    """FR9.2 — three-band classification."""
    c   = cfg or DEFAULT_CFG
    agg = c["scoring"]["tier_aggressive_max"]
    mod = c["scoring"]["tier_moderate_max"]
    if score <= agg: return "Aggressive"
    if score <= mod: return "Moderate"
    return "Safe"


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _get_nested(d: Dict, key: str):
    cur = d
    for k in key.split("."):
        cur = cur[k]
    return cur


def load_config(path: Optional[str]) -> Dict:
    cfg = _deep_merge(DEFAULT_CFG, {})
    if path and Path(path).exists():
        if not _YAML:
            log.warning("PyYAML not installed — using defaults.")
        else:
            try:
                with open(path) as fh:
                    cfg = _deep_merge(DEFAULT_CFG, yaml.safe_load(fh) or {})
            except Exception as e:
                log.error("Config error (%s) — using defaults.", e)
                cfg = _deep_merge(DEFAULT_CFG, {})

    for k, (lo, hi) in _CFG_RANGES.items():
        try:
            v = _get_nested(cfg, k)
        except KeyError:
            continue
        if not (lo <= v <= hi):
            raise ValueError(f"Config '{k}' = {v} outside [{lo}, {hi}]")

    if cfg["scoring"]["tier_aggressive_max"] >= cfg["scoring"]["tier_moderate_max"]:
        raise ValueError("tier_aggressive_max must be < tier_moderate_max (FR9.2)")

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# SensorInterface
# ──────────────────────────────────────────────────────────────────────────────

class SensorInterface:
    def __init__(self, cfg: Dict):
        s = cfg["sensor"]
        self._max_a = s["max_accel_ms2"]
        self._max_s = s["max_steer_degs"]
        self._gap   = s["gap_threshold_s"]
        cal = s["calibration"]
        self._sa, self._sd, self._ss = cal["accel_scale"], cal["decel_scale"], cal["steer_scale"]

    def validate_and_create(self, accel, decel, steer, ts,
                            last_valid, last_ts) -> Optional[SensorSample]:
        a, d, s = accel * self._sa, decel * self._sd, steer * self._ss
        if abs(a) > self._max_a: return None
        if abs(d) > self._max_a: return None
        if abs(s) > self._max_s: return None

        gap_fill = False
        if last_ts is not None and (ts - last_ts) > self._gap:
            if last_valid:
                return SensorSample(ts, last_valid.accel, last_valid.decel,
                                    last_valid.steer_rate, True)
            gap_fill = True
        return SensorSample(ts, a, d, s, gap_fill)


# ──────────────────────────────────────────────────────────────────────────────
# BehaviorClassifier
# ──────────────────────────────────────────────────────────────────────────────

class BehaviorClassifier:
    def __init__(self, cfg: Dict):
        t = cfg["thresholds"]; s = cfg["scoring"]
        self._ta, self._tb, self._ts = t["harsh_accel_ms2"], t["harsh_brake_ms2"], t["harsh_steer_degs"]
        self._dd = t["dedup_window_s"]
        self._ws = s["window_s"]
        self._pa, self._pb, self._ps = s["penalty_accel"], s["penalty_brake"], s["penalty_steer"]
        self._events:  List[HarshEvent] = []
        self._counts   = {"harsh_accel": 0, "harsh_brake": 0, "aggressive_steer": 0}
        self._last_ts: Dict[str, float] = {}
        self._wid = 0
        self._wstart: Optional[float] = None

    def process(self, sample: SensorSample) -> List[HarshEvent]:
        if self._wstart is None: self._wstart = sample.timestamp
        self._wid = int((sample.timestamp - self._wstart) / self._ws)
        new: List[HarshEvent] = []
        checks = [
            ("harsh_accel",      sample.accel,      self._ta, lambda v, t: v > t),
            ("harsh_brake",      sample.decel,      self._tb, lambda v, t: v < t),
            ("aggressive_steer", sample.steer_rate, self._ts, lambda v, t: abs(v) > t),
        ]
        for et, val, thr, cond in checks:
            if not cond(val, thr): continue
            lt = self._last_ts.get(et)
            if lt and (sample.timestamp - lt) < self._dd: continue
            e = HarshEvent(et, sample.timestamp, val, self._wid)
            self._events.append(e); self._counts[et] += 1
            self._last_ts[et] = sample.timestamp; new.append(e)
        return new

    def compute_window_score(self, rw: float = 1.0) -> int:
        pen = sum(
            (self._pa if e.event_type == "harsh_accel" else
             self._pb if e.event_type == "harsh_brake" else self._ps)
            for e in self._events if e.window_id == self._wid
        )
        return max(0, min(100, round(100 - pen * rw)))

    def get_counts(self)              -> Dict[str, int]:   return dict(self._counts)
    def get_all_events(self)          -> List[HarshEvent]: return list(self._events)
    def get_counts_since(self, since) -> Dict[str, int]:
        c = {"harsh_accel": 0, "harsh_brake": 0, "aggressive_steer": 0}
        for e in self._events:
            if e.timestamp >= since: c[e.event_type] += 1
        return c

    def reset(self):
        self._events.clear()
        self._counts  = {"harsh_accel": 0, "harsh_brake": 0, "aggressive_steer": 0}
        self._last_ts.clear(); self._wid = 0; self._wstart = None


# ──────────────────────────────────────────────────────────────────────────────
# ScoreEngine
# ──────────────────────────────────────────────────────────────────────────────

class ScoreEngine:
    def __init__(self, cfg: Dict):
        self._d = cfg["scoring"]["ewma_decay"]
        self._score = 100.0
        self._series: List[float] = []

    def update(self, ws: int) -> float:
        self._score = self._d * self._score + (1 - self._d) * ws
        self._series.append(self._score); return self._score

    def get(self)    -> int:         return max(0, min(100, round(self._score)))
    def series(self) -> List[float]: return list(self._series)
    def reset(self):                 self._score = 100.0; self._series.clear()


# ──────────────────────────────────────────────────────────────────────────────
# AlertBus / RiskAggregator
# ──────────────────────────────────────────────────────────────────────────────

class AlertBus:
    def __init__(self):
        self._state: Dict = {}; self._lock = threading.Lock()
        self._cbs:   List[Callable[[Dict], None]] = []

    def notify(self, state: Dict):
        with self._lock: self._state.update(state)
        snap = self.get_state()
        for cb in self._cbs:
            try: cb(snap)
            except Exception as e: log.warning("AlertBus: %s", e)

    def get_state(self) -> Dict:
        with self._lock: return dict(self._state)

    def register(self, cb): self._cbs.append(cb)


class RiskAggregator:
    _KEYS = ("drowsiness_alert", "distraction_alert")

    def __init__(self, cfg: Dict):
        self._w = cfg["risk"]["impairment_weight"]
        self._state: Dict = {}; self._lock = threading.Lock()

    def update(self, state: Dict):
        with self._lock: self._state.update(state)

    def weight(self) -> float:
        with self._lock: s = dict(self._state)
        return self._w if any(s.get(k) for k in self._KEYS) else 1.0


# ──────────────────────────────────────────────────────────────────────────────
# FeedbackEngine
# ──────────────────────────────────────────────────────────────────────────────

class FeedbackEngine:
    def __init__(self, cfg: Dict):
        fb = cfg["feedback"]
        self._interval  = fb["periodic_interval_s"]
        self._trend_thr = fb["trend_threshold"]
        self._recs      = cfg["recommendations"]
        self._last_ts:  Optional[float] = None

    def tick(self, score, tier, counts, elapsed, rw, jstart, sample_ts, clf):
        if self._last_ts is None: self._last_ts = jstart
        if (sample_ts - self._last_ts) < self._interval: return None
        since  = clf.get_counts_since(self._last_ts)
        report = {"report_type": "periodic", "generated_at": sample_ts,
                  "session_score": score, "tier": tier, "event_counts": since,
                  "improvement_tip": self._tip(since),
                  "high_priority_alert": tier == "Aggressive", "risk_weight": rw}
        self._last_ts = sample_ts
        return report

    def summary(self, score, tier, events, jstart, jend, dist_km,
                rolling_avg, series, cfg) -> Dict:
        dur    = (jend - jstart) / 60
        counts = {"harsh_accel": 0, "harsh_brake": 0, "aggressive_steer": 0}
        for e in events: counts[e.event_type] += 1
        top5   = sorted(events, key=lambda e: abs(e.magnitude), reverse=True)[:5]
        top_ev = [{"type": e.event_type,
                   "timestamp": datetime.fromtimestamp(e.timestamp, tz=timezone.utc).isoformat(),
                   "magnitude": round(e.magnitude, 3)} for e in top5]
        recs   = self._build_recs(counts)
        trend  = None
        if rolling_avg is not None:
            d = score - rolling_avg; thr = cfg["feedback"]["trend_threshold"]
            trend = "improving" if d >= thr else ("regressing" if d <= -thr else None)
        return {"report_type": "journey_summary",
                "journey_start": datetime.fromtimestamp(jstart, tz=timezone.utc).isoformat(),
                "journey_end":   datetime.fromtimestamp(jend,   tz=timezone.utc).isoformat(),
                "duration_minutes": round(dur, 2), "distance_km": round(dist_km, 3),
                "final_score": score, "tier": tier, "event_counts": counts,
                "top_events": top_ev, "recommendations": recs,
                "rolling_avg_score": round(rolling_avg, 2) if rolling_avg else None,
                "trend": trend, "score_series": [round(s, 3) for s in series]}

    def _tip(self, counts):
        if not any(counts.values()): return self._recs["default"]
        m = max(counts.values())
        if m == 0: return self._recs["default"]
        tied = [k for k, v in counts.items() if v == m]
        for p in ["harsh_brake", "harsh_accel", "aggressive_steer"]:
            if p in tied: return self._recs.get(p, self._recs["default"])
        return self._recs.get(max(counts, key=counts.get), self._recs["default"])

    def _build_recs(self, counts):
        recs = []
        for et, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            if counts[et] > 0 and len(recs) < 3 and et in self._recs:
                recs.append(self._recs[et])
        return recs or [self._recs["default"]]

    def reset(self): self._last_ts = None


# ──────────────────────────────────────────────────────────────────────────────
# JSONLogger / HistoryStore
# ──────────────────────────────────────────────────────────────────────────────

class JSONLogger:
    def _write(self, data, path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as fh: json.dump(data, fh, indent=2)
            log.info("Saved → %s", path); return path
        except Exception as e:
            log.error("Write failed: %s", e); return None

    def write_report(self, r, out_dir):
        ts = datetime.fromtimestamp(r.get("generated_at", time.time()), tz=timezone.utc)
        return self._write(r, Path(out_dir) / ts.strftime("periodic_%Y-%m-%dT%H-%M-%S.json"))

    def write_summary(self, s, out_dir):
        raw  = s.get("journey_start", datetime.now(tz=timezone.utc).isoformat())
        safe = raw.replace(":", "-").replace("+", "").replace(" ", "T")
        return self._write(s, Path(out_dir) / f"{safe}.json")


class HistoryStore:
    def __init__(self, cfg):
        fb = cfg["feedback"]
        self._win  = fb["rolling_window"]
        self._file = fb["history_file"]
        self._dir  = fb["output_dir"]

    def _path(self): return Path(self._dir) / self._file

    def load_scores(self):
        p = self._path()
        if not p.exists(): return []
        scores = []
        try:
            with open(p) as fh:
                for line in fh:
                    line = line.strip()
                    if not line: continue
                    try:
                        e = json.loads(line)
                        if "final_score" in e: scores.append(float(e["final_score"]))
                    except: pass
        except: pass
        return scores[-self._win:]

    def avg(self, scores): return sum(scores) / len(scores) if scores else None

    def append(self, summary):
        try:
            p = self._path(); p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as fh: fh.write(json.dumps(summary) + "\n")
        except Exception as e: log.error("History error: %s", e)

    def get_all(self):
        p = self._path()
        if not p.exists(): return []
        entries = []
        try:
            with open(p) as fh:
                for line in fh:
                    if line.strip():
                        try: entries.append(json.loads(line.strip()))
                        except: pass
        except: pass
        return entries


# ──────────────────────────────────────────────────────────────────────────────
# OverlayRenderer  — beautiful dark-glass HUD
# ──────────────────────────────────────────────────────────────────────────────

class OverlayRenderer:
    """
    Premium dark-glass heads-up display.

    Layout (left → right):
      ┌──────────────┐  ┌─────────────────────┐  ┌─────────────────┐
      │  Score ring  │  │  Sensor gauges       │  │  Event ticker   │
      │  Tier badge  │  │  Journey totals      │  │                 │
      │  Meta stats  │  │                      │  │                 │
      └──────────────┘  └─────────────────────┘  └─────────────────┘
      ═══════════════ Risk-weight banner (bottom) ════════════════════
                   ── Periodic-report toast (centre) ──
    """

    # BGR colour palette
    _SAFE  = ( 80, 210,  80)   # teal-green
    _MOD   = (  0, 185, 245)   # amber-yellow
    _AGG   = ( 45,  45, 220)   # vivid red
    _WHITE = (230, 230, 230)
    _DIM   = (120, 120, 130)
    _DARK  = ( 10,  12,  16)
    _PANEL = ( 18,  20,  26)

    _FD  = cv2.FONT_HERSHEY_DUPLEX
    _FM  = cv2.FONT_HERSHEY_SIMPLEX

    _TICKER_MAX = 7

    def __init__(self):
        self._ticker: deque = deque(maxlen=self._TICKER_MAX)
        self._toast_score: Optional[int]  = None
        self._toast_tier:  Optional[str]  = None
        self._toast_tip:   Optional[str]  = None
        self._toast_until: float = 0.0
        self._score_disp:  float = 100.0

    # ── colour ───────────────────────────────────────────────────────────────

    def _tc(self, tier: str) -> Tuple:
        return {"Safe": self._SAFE, "Moderate": self._MOD,
                "Aggressive": self._AGG}.get(tier, self._WHITE)

    # ── drawing primitives ────────────────────────────────────────────────────

    @staticmethod
    def _alpha_rect(img: np.ndarray, x1, y1, x2, y2,
                    fill, alpha=0.60, r=10):
        ov = img.copy()
        cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), fill, -1)
        cv2.rectangle(ov, (x1, y1+r), (x2, y2-r), fill, -1)
        for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(ov, (cx,cy), r, fill, -1)
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)

    @staticmethod
    def _put(img, text, x, y, font, scale, color, thick=1):
        cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

    @staticmethod
    def _text_size(text, font, scale, thick=1):
        (w, h), _ = cv2.getTextSize(text, font, scale, thick)
        return w, h

    # ── score arc ────────────────────────────────────────────────────────────

    def _draw_ring(self, img, score, tier, cx, cy, r=56):
        color = self._tc(tier)
        # Track arc
        cv2.ellipse(img, (cx,cy), (r,r), -225, 0, 270, (45,47,52), 10, cv2.LINE_AA)
        # Score arc
        span = max(1, int(270 * score / 100))
        cv2.ellipse(img, (cx,cy), (r,r), -225, 0, span, color, 10, cv2.LINE_AA)
        # Tip glow
        ar = math.radians(-225 + span)
        tx, ty = int(cx + r*math.cos(ar)), int(cy + r*math.sin(ar))
        cv2.circle(img, (tx,ty), 7, color, -1, cv2.LINE_AA)
        cv2.circle(img, (tx,ty), 11, tuple(max(0,c-60) for c in color), 1, cv2.LINE_AA)
        # Animated score number
        self._score_disp += (score - self._score_disp) * 0.18
        s = str(int(round(self._score_disp)))
        sw, sh = self._text_size(s, self._FD, 1.4, 2)
        self._put(img, s, cx - sw//2, cy + sh//2, self._FD, 1.4, color, 2)
        self._put(img, "/ 100", cx - 18, cy + sh//2 + 20, self._FM, 0.36, self._DIM)

    # ── gauge bar ─────────────────────────────────────────────────────────────

    def _gauge(self, img, label, value, max_v, thresh, x, y, w=175, h=9):
        ratio  = min(1.0, abs(value) / max_v)
        harsh  = abs(value) > thresh
        fill_c = self._AGG if harsh else self._SAFE
        # Track
        cv2.rectangle(img, (x, y), (x+w, y+h), (38,40,46), -1)
        # Fill
        fw = int(w * ratio)
        if fw > 0: cv2.rectangle(img, (x, y), (x+fw, y+h), fill_c, -1)
        # Threshold tick
        tx2 = x + int(w * thresh / max_v)
        cv2.line(img, (tx2, y-4), (tx2, y+h+4), (180,180,180), 1, cv2.LINE_AA)
        # Label
        self._put(img, label, x, y-5, self._FM, 0.36, self._DIM)
        # Value
        vs = f"{value:+.2f}"
        vc = self._AGG if harsh else self._WHITE
        self._put(img, vs, x+w+6, y+h, self._FM, 0.37, vc)
        if harsh:
            self._put(img, "HARSH", x+w+52, y+h, self._FM, 0.33, self._AGG)

    # ── event ticker ──────────────────────────────────────────────────────────

    def push_event(self, etype, magnitude, ts):
        icons = {"harsh_accel": "ACCEL", "harsh_brake": "BRAKE",
                 "aggressive_steer": "STEER"}
        t_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        self._ticker.append(f"{icons.get(etype,'EVT')}  {magnitude:+.2f}  {t_str}")

    def _draw_ticker(self, img, x, y):
        self._put(img, "HARSH EVENT LOG", x, y-6, self._FM, 0.36, self._DIM)
        items = list(self._ticker)
        for i, item in enumerate(reversed(items)):
            alpha = max(0.25, 1.0 - i * 0.14)
            c = tuple(int(v * alpha) for v in self._AGG)
            self._put(img, item, x, y + 16 + i * 18, self._FM, 0.37, c)

    # ── periodic toast ───────────────────────────────────────────────────────

    def show_toast(self, score, tier, tip):
        self._toast_score = score; self._toast_tier = tier
        self._toast_tip   = tip;   self._toast_until = time.time() + 9.0

    def _draw_toast(self, img):
        if time.time() > self._toast_until or self._toast_tip is None: return
        H, W = img.shape[:2]
        tw, th = 480, 94
        tx = W//2 - tw//2; ty = H - th - 28
        c = self._tc(self._toast_tier or "Safe")
        self._alpha_rect(img, tx, ty, tx+tw, ty+th, self._DARK, 0.84)
        cv2.line(img, (tx, ty), (tx+tw, ty), c, 2, cv2.LINE_AA)
        self._put(img, "15-MIN REPORT  (FR9.3)", tx+14, ty+20, self._FM, 0.42, c)
        self._put(img, f"Score: {self._toast_score}  [{self._toast_tier}]",
                  tx+14, ty+42, self._FM, 0.46, self._WHITE)
        # Wrap tip
        words = (self._toast_tip or "").split()
        line, lines = "", []
        for w in words:
            test = (line+" "+w).strip()
            tw2, _ = self._text_size(test, self._FM, 0.37)
            if tw2 > tw - 30: lines.append(line); line = w
            else:             line = test
        if line: lines.append(line)
        for i, l in enumerate(lines[:2]):
            self._put(img, l, tx+14, ty+62+i*17, self._FM, 0.37, self._DIM)

    # ── impairment banner ────────────────────────────────────────────────────

    def _draw_impairment(self, img, rw):
        if rw <= 1.0: return
        H, W = img.shape[:2]
        self._alpha_rect(img, 0, H-42, W, H-10, (28,28,75), 0.78)
        pulse = self._AGG if int(time.time()*2) % 2 == 0 else (90, 90, 210)
        msg = f"  IMPAIRMENT DETECTED  |  Risk Weight: {rw:.2f}x  |  Score penalties amplified  (FR9.4)"
        self._put(img, msg, 10, H-22, self._FM, 0.44, pulse)

    # ── main render ───────────────────────────────────────────────────────────

    def render(self, frame: np.ndarray, score: int, tier: str,
               accel: float, decel: float, steer: float,
               counts: Dict, rw: float, elapsed_s: float,
               dist_km: float, rolling_avg: Optional[float],
               new_events: List[HarshEvent]) -> np.ndarray:

        for e in new_events:
            self.push_event(e.event_type, e.magnitude, e.timestamp)

        out = frame.copy()
        H, W = out.shape[:2]
        color = self._tc(tier)

        # ── LEFT PANEL ───────────────────────────────────────────────────────
        PW, PH = 215, 320
        self._alpha_rect(out, 8, 8, 8+PW, 8+PH, self._PANEL, 0.78, r=12)
        cv2.rectangle(out, (8,8), (8+PW, 8+PH), color, 1, cv2.LINE_AA)
        # Thin top accent line
        cv2.rectangle(out, (8,8), (8+PW, 11), color, -1)

        self._put(out, "NEURODRIVE  |  MOD-9", 18, 28, self._FM, 0.36, self._DIM)
        self._put(out, "DRIVING STYLE", 18, 50, self._FD, 0.52, self._WHITE, 1)

        # Score ring
        ring_cx = 8 + PW//2; ring_cy = 140
        self._draw_ring(out, score, tier, ring_cx, ring_cy, r=54)

        # Tier pill
        tier_s = tier.upper()
        tsw, tsh = self._text_size(tier_s, self._FD, 0.56, 2)
        bx = ring_cx - tsw//2 - 12; by = ring_cy + 74
        self._alpha_rect(out, bx, by, bx+tsw+24, by+tsh+12, color, 0.28, r=7)
        cv2.rectangle(out, (bx,by), (bx+tsw+24, by+tsh+12), color, 1, cv2.LINE_AA)
        self._put(out, tier_s, bx+12, by+tsh+5, self._FD, 0.56, color, 2)

        # Meta stats
        my = by + tsh + 36
        em = int(elapsed_s // 60); es = int(elapsed_s % 60)
        stats = [
            ("ELAPSED",   f"{em:02d}:{es:02d}"),
            ("DISTANCE",  f"{dist_km:.3f} km"),
            ("7-TRIP AVG", f"{rolling_avg:.0f}" if rolling_avg else "---"),
            ("RISK WT",   f"{rw:.2f}x"),
        ]
        for lbl, val in stats:
            self._put(out, lbl, 20, my,    self._FM, 0.33, self._DIM)
            vc = self._AGG if (lbl == "RISK WT" and rw > 1.0) else self._WHITE
            self._put(out, val, 20, my+16, self._FM, 0.43, vc)
            my += 36

        # FR label bottom
        fr_s = f"FR9.2  [{score}/100]"
        fw2, _ = self._text_size(fr_s, self._FM, 0.33)
        self._put(out, fr_s, 8+PW-fw2-6, 8+PH-7, self._FM, 0.33, color)

        # ── CENTRE-TOP: SENSOR GAUGES ─────────────────────────────────────────
        gx = 235; gy = 14; gw = 180
        self._alpha_rect(out, gx-10, gy, gx+gw+120, gy+115, self._PANEL, 0.72, r=10)
        cv2.rectangle(out, (gx-10, gy), (gx+gw+120, gy+115), (55,57,64), 1, cv2.LINE_AA)
        cv2.rectangle(out, (gx-10, gy), (gx+gw+120, gy+4), (55,57,64), -1)

        self._put(out, "SENSOR INPUTS  (FR9.1  10 Hz)", gx, gy+18, self._FM, 0.39, self._DIM)
        self._gauge(out, "ACCEL  threshold > 2.5 m/s²",
                    accel,      9.8,  2.5,  gx, gy+40, gw)
        self._gauge(out, "BRAKE  threshold < -3.0 m/s²",
                    decel,      9.8,  3.0,  gx, gy+70, gw)
        self._gauge(out, "STEER  threshold > 15 deg/s",
                    steer,      60.0, 15.0, gx, gy+100, gw)

        # ── CENTRE-BOTTOM: JOURNEY TOTALS ─────────────────────────────────────
        ex = 235; ey = 142
        self._alpha_rect(out, ex-10, ey, ex+310, ey+100, self._PANEL, 0.72, r=10)
        cv2.rectangle(out, (ex-10, ey), (ex+310, ey+100), (55,57,64), 1, cv2.LINE_AA)
        self._put(out, "JOURNEY TOTALS", ex, ey+18, self._FM, 0.39, self._DIM)

        cards = [
            ("HARSH ACCEL", counts.get("harsh_accel",      0), self._MOD),
            ("HARSH BRAKE", counts.get("harsh_brake",       0), self._AGG),
            ("AGGR STEER",  counts.get("aggressive_steer", 0), (100, 200, 255)),
        ]
        for i, (lbl, cnt, c) in enumerate(cards):
            bx2 = ex + i * 104
            cv2.rectangle(out, (bx2, ey+28), (bx2+96, ey+92), (28,30,36), -1)
            cv2.rectangle(out, (bx2, ey+28), (bx2+96, ey+92), c, 1, cv2.LINE_AA)
            cs = str(cnt)
            csw, csh = self._text_size(cs, self._FD, 1.2, 2)
            self._put(out, cs, bx2+48-csw//2, ey+66, self._FD, 1.2, c, 2)
            self._put(out, lbl, bx2+4, ey+88, self._FM, 0.30, self._DIM)

        # ── RIGHT PANEL: EVENT TICKER ─────────────────────────────────────────
        rx = max(W - 275, 560)
        self._alpha_rect(out, rx-8, 8, W-8, 8+162, self._PANEL, 0.72, r=10)
        cv2.rectangle(out, (rx-8, 8), (W-8, 8+162), (55,57,64), 1, cv2.LINE_AA)
        cv2.rectangle(out, (rx-8, 8), (W-8, 12), (55,57,64), -1)
        self._draw_ticker(out, rx, 32)

        # ── BANNERS ───────────────────────────────────────────────────────────
        self._draw_impairment(out, rw)
        self._draw_toast(out)

        return out


# ──────────────────────────────────────────────────────────────────────────────
# DrivingStyleFeedback — Public Façade
# ──────────────────────────────────────────────────────────────────────────────

class DrivingStyleFeedback:
    """Module 9: Driving Style Feedback — full public API."""

    def __init__(self, config_path: Optional[str] = None):
        self._cfg     = load_config(config_path)
        fb            = self._cfg["feedback"]
        self._sensor  = SensorInterface(self._cfg)
        self._clf     = BehaviorClassifier(self._cfg)
        self._score   = ScoreEngine(self._cfg)
        self._bus     = AlertBus()
        self._risk    = RiskAggregator(self._cfg)
        self._fb      = FeedbackEngine(self._cfg)
        self._jlogger = JSONLogger()
        self._hist    = HistoryStore(self._cfg)
        self._hud     = OverlayRenderer()
        self._bus.register(self._risk.update)

        self._active   = False
        self._jstart:  Optional[float]        = None
        self._last_s:  Optional[SensorSample] = None
        self._last_ts: Optional[float]        = None
        self._dist_km  = 0.0
        self._out_dir  = fb["output_dir"]
        self._last_accel = self._last_decel = self._last_steer = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start_journey(self):
        """Begin a new driving session (resets all state)."""
        self._clf.reset(); self._score.reset(); self._fb.reset()
        self._active = True; self._jstart = time.time()
        self._last_s = self._last_ts = None; self._dist_km = 0.0
        log.info("Journey started  %s",
                 datetime.fromtimestamp(self._jstart, tz=timezone.utc).isoformat())

    def stop_journey(self) -> Dict:
        """End session. Returns FR9.5 summary dict (also saved to disk)."""
        if not self._active:
            log.warning("stop_journey: no active journey."); return {}
        jend = time.time(); self._active = False
        sc   = self._score.get(); tier = classify_tier(sc, self._cfg)
        past = self._hist.load_scores()
        smry = self._fb.summary(sc, tier, self._clf.get_all_events(),
                                self._jstart, jend, self._dist_km,
                                self._hist.avg(past), self._score.series(), self._cfg)
        self._jlogger.write_summary(smry, self._out_dir)
        self._hist.append(smry)
        log.info("Journey ended  score=%d (%s)  %.1f min  %.3f km",
                 sc, tier, smry["duration_minutes"], self._dist_km)
        return smry

    # ── per-sample at 10 Hz (FR9.1) ───────────────────────────────────────────

    def process_sample(self, accel: float, decel: float, steer_rate: float,
                       timestamp: Optional[float] = None,
                       speed_ms: float = 0.0) -> Dict:
        """FR9.1 — process one sensor reading. Call at 10 Hz."""
        if not self._active: return {}
        ts = timestamp if timestamp is not None else time.time()

        sample = self._sensor.validate_and_create(
            accel, decel, steer_rate, ts, self._last_s, self._last_ts)
        if sample is None: return self._snap()

        if self._last_ts is not None and speed_ms > 0:
            self._dist_km += (speed_ms * (ts - self._last_ts)) / 1000.0

        self._last_s     = sample
        self._last_ts    = ts
        self._last_accel = sample.accel
        self._last_decel = sample.decel
        self._last_steer = sample.steer_rate

        new_evts = self._clf.process(sample)
        rw       = self._risk.weight()
        ws       = self._clf.compute_window_score(rw)
        self._score.update(ws)

        sc   = self._score.get()
        tier = classify_tier(sc, self._cfg)

        report = self._fb.tick(sc, tier, self._clf.get_counts(),
                               ts - self._jstart, rw, self._jstart, ts, self._clf)
        if report:
            self._jlogger.write_report(report, self._out_dir)
            self._hud.show_toast(report["session_score"], report["tier"],
                                 report["improvement_tip"])
            log.info("Periodic report (FR9.3): score=%d tier=%s",
                     report["session_score"], report["tier"])

        return self._snap(report, new_evts)

    # ── queries ───────────────────────────────────────────────────────────────

    def get_current_score(self) -> Tuple[int, str]:
        sc = self._score.get(); return sc, classify_tier(sc, self._cfg)

    def get_journey_snapshot(self) -> Dict:
        if not self._active: return {}
        sc, tier = self.get_current_score()
        return {"status": "in_progress",
                "elapsed_min": round((time.time()-self._jstart)/60, 2),
                "current_score": sc, "tier": tier,
                "event_counts": self._clf.get_counts(),
                "distance_km": round(self._dist_km, 3),
                "risk_weight": self._risk.weight()}

    def get_score_history(self) -> List[float]: return self._score.series()

    # ── rendering ─────────────────────────────────────────────────────────────

    def render_overlay(self, frame: np.ndarray,
                       new_events: Optional[List[HarshEvent]] = None) -> np.ndarray:
        """Render premium HUD. Returns frame unchanged if no journey active."""
        if not self._active: return frame
        sc, tier = self.get_current_score()
        elapsed  = time.time() - self._jstart
        ravg     = self._hist.avg(self._hist.load_scores())
        return self._hud.render(
            frame, sc, tier,
            self._last_accel, self._last_decel, self._last_steer,
            self._clf.get_counts(), self._risk.weight(), elapsed,
            self._dist_km, ravg, new_events or [])

    # ── FR9.4 integration ─────────────────────────────────────────────────────

    def update_alert_state(self, state: Dict):
        """
        FR9.4 — push external alert state (FCW / Drowsiness / Distraction).
        Keys: drowsiness_alert (bool), distraction_alert (bool), fcw_alert (bool).
        """
        self._bus.notify(state)

    # ── internal ──────────────────────────────────────────────────────────────

    def _snap(self, report=None, new_events=None) -> Dict:
        sc, tier = self.get_current_score()
        return {"score": sc, "tier": tier,
                "event_counts": self._clf.get_counts(),
                "risk_weight": self._risk.weight(),
                "distance_km": round(self._dist_km, 3),
                "periodic_report": report,
                "new_events": new_events or []}


# ──────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(s: Dict):
    """FR9.5 — pretty-print end-of-journey summary to terminal."""
    tier = s.get("tier", "")
    sym  = {"Safe": "✅", "Moderate": "⚡", "Aggressive": "🚨"}.get(tier, "")
    c    = s.get("event_counts", {})
    recs = s.get("recommendations", [])
    ra   = s.get("rolling_avg_score"); tr = s.get("trend")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        NeuroDrive — MODULE 9 JOURNEY SUMMARY  (FR9.5)       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Start          {s.get('journey_start','N/A'):<44}║")
    print(f"║  End            {s.get('journey_end','N/A'):<44}║")
    print(f"║  Duration       {s.get('duration_minutes',0):.1f} min{'':<40}║")
    print(f"║  Distance       {s.get('distance_km',0):.3f} km{'':<40}║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Final Score    {s.get('final_score',0)}/100   {sym} {tier:<36}║")
    if ra: print(f"║  7-Trip Avg     {ra:.1f}{'':<45}║")
    if tr:
        arrow = "↑ Improving" if tr == "improving" else "↓ Regressing"
        print(f"║  Trend          {arrow:<45}║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Harsh Accel    {c.get('harsh_accel',0):<45}║")
    print(f"║  Harsh Brake    {c.get('harsh_brake',0):<45}║")
    print(f"║  Aggr. Steer    {c.get('aggressive_steer',0):<45}║")
    if recs:
        print("╠══════════════════════════════════════════════════════════════╣")
        print("║  RECOMMENDATIONS:                                            ║")
        for i, r in enumerate(recs, 1):
            chunks = [r[j:j+56] for j in range(0, len(r), 56)]
            for ci, chunk in enumerate(chunks):
                prefix = f"{i}. " if ci == 0 else "   "
                print(f"║  {prefix}{chunk:<57}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")


def _synthetic_sample(frame_idx: int, speed_kmh: float) -> Tuple[float,float,float,float]:
    t     = frame_idx / 10.0
    accel = 0.5 * math.sin(t * 0.3) + 0.2 * math.sin(t * 1.1)
    decel = -0.3 * abs(math.sin(t * 0.5)) - 0.1
    steer = 8.0 * math.sin(t * 0.7)
    if frame_idx % 80  == 0: accel =  3.2
    if frame_idx % 120 == 0: decel = -3.8
    if frame_idx % 150 == 0: steer = 22.0
    return accel, decel, steer, speed_kmh / 3.6


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="NeuroDrive Module 9 — Driving Style Feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples:\n"
                "  python driving_style_feedback.py --video test.mp4 --speed 60\n"
                "  python driving_style_feedback.py --synthetic --speed 80 --duration 60\n"))
    ap.add_argument("--video",     type=str)
    ap.add_argument("--speed",     type=float, default=60.0)
    ap.add_argument("--config",    type=str,   default=None)
    ap.add_argument("--output",    type=str,   default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--duration",  type=int,   default=60)
    args = ap.parse_args()

    dsf = DrivingStyleFeedback(config_path=args.config)
    dsf.start_journey()

    cap = writer = None
    out_path = args.output or "dsf_output/output.mp4"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            log.error("Cannot open video: %s", args.video); sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        log.info("Video: %s  %.0f fps  %dx%d  %d frames", args.video, fps, W, H, total)
    else:
        log.info("Synthetic mode — %d seconds", args.duration)

    frame_idx = 0
    interval  = 1.0 / 10.0
    next_tick = time.time()

    try:
        while True:
            if cap is not None:
                ret, frame = cap.read()
                if not ret: break
            else:
                if frame_idx >= args.duration * 10: break
                frame = np.full((720, 1280, 3), 28, dtype=np.uint8)
                cv2.rectangle(frame, (0,360), (1280,720), (18,18,18), -1)
                cv2.line(frame, (640,360),(640,720),(55,55,55),2)

            now = time.time()
            new_evts: List[HarshEvent] = []
            if now >= next_tick:
                a, d, s, spd = _synthetic_sample(frame_idx, args.speed)
                snap    = dsf.process_sample(a, d, s, speed_ms=spd)
                new_evts = snap.get("new_events", [])
                next_tick = now + interval
                frame_idx += 1

            annotated = dsf.render_overlay(frame, new_evts)
            if writer: writer.write(annotated)

            cv2.imshow("NeuroDrive — Module 9: Driving Style Feedback  [Q to quit]",
                       annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27): break

    except KeyboardInterrupt:
        pass
    finally:
        if cap:    cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()

    summary = dsf.stop_journey()
    _print_summary(summary)
    log.info("Output saved → %s", out_path)


if __name__ == "__main__":
    main()