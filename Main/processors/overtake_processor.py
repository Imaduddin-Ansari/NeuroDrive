"""
overtake_processor.py
─────────────────────
Processor wrapper for the Overtake Assistance module.

This is NOT a video-reading processor — it is a stateful coordinator that:
  • holds the overtake FSM (Finite State Machine)
  • calls OvertakeAnalyser.analyse() in a background thread when active
  • exposes a latest_result attribute for the UI to read
  • submits a rich RiskSnapshot to the LLM explainer

The UI drives the FSM via:
    processor.request_overtake()        # Overtake button pressed
    processor.set_side(OvertakeSide.X)  # indicator pressed
    processor.cancel()                  # overtake button pressed again / ESC

State machine:
    IDLE → (request_overtake) → WAITING → (set_side) → CHECKING
    CHECKING → (analysis done) → SAFE / CAUTION / UNSAFE
    Any state → (cancel / timeout) → IDLE
"""

from __future__ import annotations

import threading
import time
from typing import Optional

# ── Inline import guard so the file survives even if the OvertakeAssistance
#    package isn't on sys.path yet.  main_window.py adds the project root.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from OvertakeAssistance.overtake_logic import (
    OvertakeAnalyser, OvertakeResult, OvertakeState, OvertakeSide,
)

# How long (seconds) the system waits for the driver to pick a side
SIDE_SELECTION_TIMEOUT = 15.0
# How long (seconds) a SAFE/CAUTION/UNSAFE result stays on screen before expiring
RESULT_TTL = 12.0


class OvertakeProcessor:
    """
    Lightweight stateful coordinator — no video queue, no heavy thread.
    The analysis thread is short-lived (fires once per request).
    """

    def __init__(self):
        self._lock             = threading.Lock()
        self._state            = OvertakeState.IDLE
        self._side: Optional[OvertakeSide] = None
        self._result: Optional[OvertakeResult] = None
        self._request_time: float = 0.0
        self._result_time:  float = 0.0
        self._analyser       = OvertakeAnalyser(yolo=None)   # YOLO injected later
        self._analysis_thread: Optional[threading.Thread] = None
        self.is_running       = True

        # Callbacks — set by main_window after construction
        self.on_result_ready  = None   # callable(OvertakeResult)

    # ── Processor-level accessors (thread-safe) ───────────────────────────────

    @property
    def state(self) -> OvertakeState:
        with self._lock:
            return self._state

    @property
    def latest_result(self) -> Optional[OvertakeResult]:
        with self._lock:
            return self._result

    @property
    def active_side(self) -> Optional[OvertakeSide]:
        with self._lock:
            return self._side

    # ── FSM transitions ───────────────────────────────────────────────────────

    def request_overtake(self):
        """Called when the driver presses the Overtake button."""
        with self._lock:
            if self._state in (OvertakeState.IDLE, OvertakeState.EXPIRED,
                               OvertakeState.SAFE, OvertakeState.CAUTION,
                               OvertakeState.UNSAFE):
                self._state        = OvertakeState.WAITING
                self._side         = None
                self._result       = None
                self._request_time = time.time()
                print("[Overtake] Waiting for side selection…")
            elif self._state == OvertakeState.WAITING:
                # Second press cancels
                self._state = OvertakeState.IDLE
                self._side  = None
                print("[Overtake] Cancelled (button pressed again).")

    def set_side(self, side: OvertakeSide):
        """
        Called when the driver activates an indicator while in WAITING state.
        Immediately kicks off the analysis thread.
        """
        with self._lock:
            if self._state != OvertakeState.WAITING:
                return
            self._side  = side
            self._state = OvertakeState.CHECKING
            print(f"[Overtake] Side selected: {side.value} — starting analysis…")

        # Launch analysis in background so we don't block Tkinter
        self._analysis_thread = threading.Thread(
            target=self._run_analysis, daemon=True, name="OvertakeAnalysis"
        )
        self._analysis_thread.start()

    def cancel(self):
        with self._lock:
            self._state  = OvertakeState.IDLE
            self._side   = None
            self._result = None
        print("[Overtake] Cancelled.")

    def tick(self):
        """
        Call from the UI tick loop to handle timeouts and result expiry.
        Returns the current state so the UI can react without an extra lock.
        """
        with self._lock:
            now = time.time()

            # Waiting too long for side selection
            if (self._state == OvertakeState.WAITING and
                    now - self._request_time > SIDE_SELECTION_TIMEOUT):
                self._state = OvertakeState.IDLE
                self._side  = None
                print("[Overtake] Side-selection timeout — returning to idle.")

            # Expire old result
            if (self._state in (OvertakeState.SAFE, OvertakeState.CAUTION,
                                OvertakeState.UNSAFE) and
                    self._result_time > 0 and
                    now - self._result_time > RESULT_TTL):
                self._state = OvertakeState.IDLE
                print("[Overtake] Result expired.")

            return self._state

    # ── YOLO injection ────────────────────────────────────────────────────────

    def inject_yolo(self, yolo_model):
        """Inject a loaded YOLO model so the front-frame check can run."""
        self._analyser = OvertakeAnalyser(yolo=yolo_model)

    # ── Background analysis ───────────────────────────────────────────────────

    def _run_analysis(self):
        """Background thread — reads processor state and runs the analyser."""
        # We need references to the live processors.  They are set from
        # main_window after the processors are started.
        left_bsp  = getattr(self, '_left_bsp_proc',   None)
        right_bsp = getattr(self, '_right_bsp_proc',  None)
        fcw       = getattr(self, '_fcw_proc',         None)
        front_frm = getattr(self, '_get_front_frame', None)
        ego_kmph  = getattr(self, '_ego_kmph',         50.0)

        with self._lock:
            side = self._side

        if side is None:
            return

        # Grab the latest front frame (callable provided by main_window)
        front_frame = None
        if callable(front_frm):
            try:
                front_frame = front_frm()
            except Exception:
                pass

        try:
            result = self._analyser.analyse(
                side         = side,
                left_bsp_proc  = left_bsp,
                right_bsp_proc = right_bsp,
                fcw_proc       = fcw,
                front_frame    = front_frame,
                ego_speed_kmh  = ego_kmph,
            )
        except Exception as e:
            print(f"[Overtake] Analysis error: {e}")
            with self._lock:
                self._state = OvertakeState.IDLE
            return

        with self._lock:
            self._result      = result
            self._state       = result.state
            self._result_time = time.time()

        print(f"[Overtake] Result: {result.state.value} — {result.reason}")

        if callable(self.on_result_ready):
            try:
                self.on_result_ready(result)
            except Exception:
                pass

    # ── Convenience: attach processor references ──────────────────────────────

    def attach_processors(self, left_bsp, right_bsp, fcw,
                          get_front_frame_fn, ego_kmph_fn):
        """Called by main_window once processors are started."""
        self._left_bsp_proc    = left_bsp
        self._right_bsp_proc   = right_bsp
        self._fcw_proc         = fcw
        self._get_front_frame  = get_front_frame_fn   # callable → np.ndarray | None
        self._ego_kmph         = 50.0                 # default; updated each tick
        self._ego_kmph_fn      = ego_kmph_fn          # callable → float

    def update_ego_speed(self):
        if hasattr(self, '_ego_kmph_fn') and callable(self._ego_kmph_fn):
            self._ego_kmph = self._ego_kmph_fn()

    def stop(self):
        self.is_running = False