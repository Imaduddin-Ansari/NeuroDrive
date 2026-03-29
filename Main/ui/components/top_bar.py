"""Top bar component — with Overtake Assistance button + label_override for OFF states"""
import tkinter as tk
import time
from utils.constants import COLORS, UI_DIMENSIONS

_DIM = '#666666'   # greyed-out colour for disabled indicators

# Overtake button — neutral idle, subtle state colours (no eye-searing brights)
_OT_IDLE    = '#252525'   # same as bg_lighter — blends with indicator buttons
_OT_WAIT    = '#3a3000'   # muted amber
_OT_CHECK   = '#002040'   # muted blue
_OT_SAFE    = '#1a3a1a'   # muted green (NOT neon)
_OT_CAUTION = '#3a2800'   # muted orange
_OT_UNSAFE  = '#3a0000'   # muted red


class TopBar(tk.Frame):
    """Top bar with status indicators, turn signals, overtake button and settings."""

    def __init__(self, parent, settings_callback, indicator_callback=None,
                 overtake_callback=None):
        super().__init__(
            parent,
            bg=COLORS['bg_light'],
            height=UI_DIMENSIONS['top_bar_height'],
        )
        self.pack_propagate(False)
        self.settings_callback  = settings_callback
        self.indicator_callback = indicator_callback
        self.overtake_callback  = overtake_callback   # NEW
        self._create_ui()

    def _create_ui(self):
        # ── Title ─────────────────────────────────────────────────────────────
        tk.Label(self, text="NeuroDrive",
                 font=("Helvetica", 22, "bold"),
                 fg=COLORS['accent'], bg=COLORS['bg_light'],
                 ).pack(side='left', padx=20)

        # ── System status dot ─────────────────────────────────────────────────
        sf = tk.Frame(self, bg=COLORS['bg_light'])
        sf.pack(side='left', padx=10)
        tk.Label(sf, text="●", font=("Helvetica", 20),
                 fg=COLORS['success'], bg=COLORS['bg_light']).pack(side='left')
        tk.Label(sf, text="System Active", font=("Helvetica", 12),
                 fg=COLORS['text_tertiary'], bg=COLORS['bg_light']).pack(side='left', padx=5)

        # ── FCW ───────────────────────────────────────────────────────────────
        self.fcw_frame = tk.Frame(self, bg=COLORS['bg_light'])
        self.fcw_frame.pack(side='left', padx=8)
        self.fcw_indicator = tk.Label(self.fcw_frame, text="●",
                                      font=("Helvetica", 22),
                                      fg=COLORS['success'], bg=COLORS['bg_light'])
        self.fcw_indicator.pack(side='left')
        self.fcw_label = tk.Label(self.fcw_frame, text="FCW: SAFE",
                                  font=("Helvetica", 11, "bold"),
                                  fg=COLORS['success'], bg=COLORS['bg_light'])
        self.fcw_label.pack(side='left', padx=4)

        # ── Lane ──────────────────────────────────────────────────────────────
        self.lane_frame = tk.Frame(self, bg=COLORS['bg_light'])
        self.lane_frame.pack(side='left', padx=8)
        self.lane_indicator = tk.Label(self.lane_frame, text="●",
                                       font=("Helvetica", 22),
                                       fg=COLORS['success'], bg=COLORS['bg_light'])
        self.lane_indicator.pack(side='left')
        self.lane_label = tk.Label(self.lane_frame, text="LANE: ✓",
                                   font=("Helvetica", 11, "bold"),
                                   fg=COLORS['success'], bg=COLORS['bg_light'])
        self.lane_label.pack(side='left', padx=4)

        # ── Traffic Signs ─────────────────────────────────────────────────────
        self.ts_frame = tk.Frame(self, bg=COLORS['bg_light'])
        self.ts_frame.pack(side='left', padx=8)
        self.ts_icon = tk.Label(self.ts_frame, text="🚦",
                                font=("Helvetica", 20),
                                fg=COLORS['success'], bg=COLORS['bg_light'])
        self.ts_icon.pack(side='left')
        self.ts_container = tk.Frame(self.ts_frame, bg=COLORS['bg_light'])
        self.ts_container.pack(side='left', padx=4)
        self.ts_labels = []
        for _ in range(3):
            lbl = tk.Label(self.ts_container, text="",
                           font=("Helvetica", 10, "bold"),
                           fg=COLORS['text_tertiary'], bg=COLORS['bg_light'])
            lbl.pack(anchor='w')
            self.ts_labels.append(lbl)

        # ── Blind Spot ────────────────────────────────────────────────────────
        self.bsp_frame = tk.Frame(self, bg=COLORS['bg_light'])
        self.bsp_frame.pack(side='left', padx=8)
        self.bsp_indicator = tk.Label(self.bsp_frame, text="●",
                                      font=("Helvetica", 22),
                                      fg=COLORS['success'], bg=COLORS['bg_light'])
        self.bsp_indicator.pack(side='left')
        self.bsp_label = tk.Label(self.bsp_frame, text="BLIND: CLEAR",
                                  font=("Helvetica", 11, "bold"),
                                  fg=COLORS['success'], bg=COLORS['bg_light'])
        self.bsp_label.pack(side='left', padx=4)

        # ── Priority Rules ────────────────────────────────────────────────────
        self.rules_frame = tk.Frame(self, bg=COLORS['bg_light'])
        self.rules_frame.pack(side='left', padx=8)
        self.rules_indicator = tk.Label(self.rules_frame, text="●",
                                        font=("Helvetica", 22),
                                        fg=COLORS['success'], bg=COLORS['bg_light'])
        self.rules_indicator.pack(side='left')
        self.rules_label = tk.Label(self.rules_frame, text="RULES: OK",
                                    font=("Helvetica", 11, "bold"),
                                    fg=COLORS['success'], bg=COLORS['bg_light'])
        self.rules_label.pack(side='left', padx=4)

        # ── Driver Distraction ────────────────────────────────────────────────
        self.driver_frame = tk.Frame(self, bg=COLORS['bg_light'])
        self.driver_frame.pack(side='left', padx=8)
        self.driver_indicator = tk.Label(self.driver_frame, text="●",
                                         font=("Helvetica", 22),
                                         fg=COLORS['success'], bg=COLORS['bg_light'])
        self.driver_indicator.pack(side='left')
        self.driver_label = tk.Label(self.driver_frame, text="DRIVER: OK",
                                     font=("Helvetica", 11, "bold"),
                                     fg=COLORS['success'], bg=COLORS['bg_light'])
        self.driver_label.pack(side='left', padx=4)

        # ──────────────────────────────────────────────────────────────────────
        # RIGHT SIDE  (packed right-to-left: Settings | Indicators | Overtake)
        # ──────────────────────────────────────────────────────────────────────

        # ── Settings ──────────────────────────────────────────────────────────
        tk.Button(
            self, text="⚙ Settings",
            font=("Helvetica", 12),
            bg=COLORS['accent'], fg="#000000",
            activebackground=COLORS['accent_dark'], activeforeground="#000000",
            relief='flat', padx=18, pady=8, cursor="hand2",
            command=self.settings_callback,
        ).pack(side='right', padx=20, pady=10)

        # ── Turn indicators ───────────────────────────────────────────────────
        self.ind_frame = tk.Frame(self, bg=COLORS['bg_light'])
        self.ind_frame.pack(side='right', padx=4)

        self.left_ind_btn = tk.Button(
            self.ind_frame, text="◀ L",
            font=("Helvetica", 11, "bold"),
            bg=COLORS['bg_lighter'], fg=COLORS['text_tertiary'],
            activebackground=COLORS['warning'], activeforeground="#000000",
            relief='flat', padx=12, pady=6, cursor="hand2",
            command=lambda: self._toggle_indicator('left'),
        )
        self.left_ind_btn.pack(side='left', padx=2)

        self.right_ind_btn = tk.Button(
            self.ind_frame, text="R ▶",
            font=("Helvetica", 11, "bold"),
            bg=COLORS['bg_lighter'], fg=COLORS['text_tertiary'],
            activebackground=COLORS['warning'], activeforeground="#000000",
            relief='flat', padx=12, pady=6, cursor="hand2",
            command=lambda: self._toggle_indicator('right'),
        )
        self.right_ind_btn.pack(side='left', padx=2)

        self.left_indicator_active  = False
        self.right_indicator_active = False

        # ── Overtake Assistance button ────────────────────────────────────────
        self._ot_state_label_text = tk.StringVar(value="⇉ OVERTAKE")
        self.overtake_btn = tk.Button(
            self,
            textvariable=self._ot_state_label_text,
            font=("Helvetica", 11, "bold"),
            bg=COLORS['bg_lighter'], fg=COLORS['text_secondary'],
            activebackground=COLORS['accent_dark'], activeforeground="#000000",
            relief='flat', padx=12, pady=6, cursor="hand2",
            command=self._on_overtake_pressed,
        )
        self.overtake_btn.pack(side='right', padx=2, pady=10)

        # Overtake status sub-label (shown below the button text)
        self.overtake_sublabel = tk.Label(
            self, text="",
            font=("Helvetica", 9),
            fg=COLORS['text_tertiary'], bg=COLORS['bg_light'],
        )
        # We DON'T pack it here — it appears dynamically when needed

    # ──────────────────────────────────────────────────────────────────────────
    # Overtake button handler
    # ──────────────────────────────────────────────────────────────────────────

    def _on_overtake_pressed(self):
        if callable(self.overtake_callback):
            self.overtake_callback()

    # ──────────────────────────────────────────────────────────────────────────
    # Public: update overtake button appearance from main_window tick
    # ──────────────────────────────────────────────────────────────────────────

    def update_overtake_status(self, state_str: str, sub: str = ""):
        """
        Parameters
        ──────────
        state_str : one of 'idle' | 'waiting' | 'checking' |
                    'safe' | 'caution' | 'unsafe'
        sub       : optional short sub-label (e.g. side name, verdict)
        """
        cfg = {
            # state       label text               bg             fg (text)
            'idle':     ("⇉ OVERTAKE",           COLORS['bg_lighter'], COLORS['text_secondary']),
            'waiting':  ("⇉ PICK SIDE…",          _OT_WAIT,            COLORS['warning']),
            'checking': ("⇉ CHECKING…",           _OT_CHECK,           '#7ec8e3'),
            'safe':     ("⇉ SAFE TO GO",          _OT_SAFE,            COLORS['success']),
            'caution':  ("⇉ CAUTION",             _OT_CAUTION,         COLORS['warning']),
            'unsafe':   ("⇉ DO NOT OVERTAKE",     _OT_UNSAFE,          COLORS['error_light']),
            'expired':  ("⇉ OVERTAKE",            COLORS['bg_lighter'], COLORS['text_secondary']),
        }
        txt, bg, fg = cfg.get(state_str, cfg['idle'])
        self._ot_state_label_text.set(txt)
        self.overtake_btn.config(bg=bg, fg=fg,
                                 activebackground=bg,
                                 activeforeground=fg)
        # Sub-label (e.g. "LEFT — clear / UNSAFE: vehicle 8 m")
        if sub:
            self.overtake_sublabel.config(
                text=sub,
                fg=(COLORS['success'] if state_str == 'safe' else
                    COLORS['warning'] if state_str == 'caution' else
                    COLORS['error']   if state_str == 'unsafe'  else
                    '#ffd700'         if state_str == 'waiting' else
                    COLORS['text_tertiary']),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Turn indicator toggle
    # ──────────────────────────────────────────────────────────────────────────

    def _toggle_indicator(self, side):
        if side == 'left':
            self.left_indicator_active = not self.left_indicator_active
            if self.left_indicator_active:
                self.left_ind_btn.config(bg=COLORS['warning'], fg="#000000")
                self.right_indicator_active = False
                self.right_ind_btn.config(bg=COLORS['bg_lighter'], fg=COLORS['text_tertiary'])
            else:
                self.left_ind_btn.config(bg=COLORS['bg_lighter'], fg=COLORS['text_tertiary'])
        else:
            self.right_indicator_active = not self.right_indicator_active
            if self.right_indicator_active:
                self.right_ind_btn.config(bg=COLORS['warning'], fg="#000000")
                self.left_indicator_active = False
                self.left_ind_btn.config(bg=COLORS['bg_lighter'], fg=COLORS['text_tertiary'])
            else:
                self.right_ind_btn.config(bg=COLORS['bg_lighter'], fg=COLORS['text_tertiary'])

        if self.indicator_callback:
            self.indicator_callback(self.left_indicator_active,
                                    self.right_indicator_active)

    # ──────────────────────────────────────────────────────────────────────────
    # Existing update methods (unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    def update_fcw_status(self, critical=False, offline=False, label_override=None):
        if label_override:
            self.fcw_indicator.config(fg=_DIM)
            self.fcw_label.config(text=label_override, fg=_DIM)
        elif offline:
            self.fcw_indicator.config(fg=COLORS['success'])
            self.fcw_label.config(text="FCW: SAFE ✓", fg=COLORS['success'])
        elif critical:
            c = COLORS['error'] if (int(time.time() * 4) % 2 == 0) else COLORS['error_light']
            self.fcw_indicator.config(fg=c)
            self.fcw_label.config(text="⚠ COLLISION!", fg=c)
        else:
            self.fcw_indicator.config(fg=COLORS['success'])
            self.fcw_label.config(text="FCW: SAFE ✓", fg=COLORS['success'])

    def update_lane_status(self, warning=False, direction='F', offline=False, label_override=None):
        if label_override:
            self.lane_indicator.config(fg=_DIM)
            self.lane_label.config(text=label_override, fg=_DIM)
        elif offline:
            self.lane_indicator.config(fg=COLORS['success'])
            self.lane_label.config(text="LANE: ✓", fg=COLORS['success'])
        elif warning:
            c = COLORS['error'] if (int(time.time() * 3) % 2 == 0) else COLORS['warning']
            self.lane_indicator.config(fg=c)
            sym = "←" if direction == 'L' else "→" if direction == 'R' else "↑"
            self.lane_label.config(text=f"⚠ DRIFT {sym}", fg=c)
        else:
            self.lane_indicator.config(fg=COLORS['success'])
            sym = "←" if direction == 'L' else "→" if direction == 'R' else "↑"
            self.lane_label.config(text=f"LANE: {sym} ✓", fg=COLORS['success'])

    def update_traffic_signs(self, sign_history, current_time, label_override=None):
        if label_override:
            self.ts_icon.config(text="🚦", fg=_DIM)
            for lbl in self.ts_labels:
                lbl.config(text="", fg=_DIM)
            self.ts_labels[0].config(text=label_override, fg=_DIM)
            return

        if sign_history:
            high_conf = max(s['confidence'] for s in sign_history)
            if high_conf >= 0.90:
                c = COLORS['error'] if (int(current_time * 3) % 2 == 0) else COLORS['warning']
                self.ts_icon.config(text="⚠", fg=c)
            elif high_conf >= 0.70:
                self.ts_icon.config(text="🚦", fg=COLORS['warning'])
            else:
                self.ts_icon.config(text="🚦", fg="#cccc00")

            for i, lbl in enumerate(self.ts_labels):
                if i < len(sign_history):
                    s    = sign_history[i]
                    age  = current_time - s['last_seen']
                    name = s['name'][:15] + ("..." if len(s['name']) > 18 else "")
                    if s['confidence'] >= 0.90:
                        c = COLORS['error'] if age < 0.5 else COLORS['error_light']
                        lbl.config(text=f"⚠ {name}", fg=c)
                    elif s['confidence'] >= 0.70:
                        lbl.config(text=name, fg=COLORS['warning'])
                    else:
                        lbl.config(text=name, fg="#cccc00")
                else:
                    lbl.config(text="", fg=COLORS['text_tertiary'])
        else:
            self.ts_icon.config(text="🚦", fg=COLORS['success'])
            for lbl in self.ts_labels:
                lbl.config(text="", fg=COLORS['text_tertiary'])
            self.ts_labels[0].config(text="NO SIGNS", fg=COLORS['text_tertiary'])

    def update_blindspot_status(self, detected=False, side='unknown',
                                count=0, distance=None, label_override=None):
        if label_override:
            self.bsp_indicator.config(fg=_DIM)
            self.bsp_label.config(text=label_override, fg=_DIM)
        elif detected:
            c = COLORS['error'] if (int(time.time() * 2) % 2 == 0) else COLORS['error_light']
            self.bsp_indicator.config(fg=c)
            st = side.upper() if side != 'unknown' else 'VEHICLE'
            self.bsp_label.config(text=f"⚠ {st} ({count})", fg=c)
        else:
            self.bsp_indicator.config(fg=COLORS['success'])
            self.bsp_label.config(text="BLIND: CLEAR ✓", fg=COLORS['success'])

    def update_priority_rules(self, active_rules, label_override=None):
        if label_override:
            self.rules_indicator.config(fg=_DIM)
            self.rules_label.config(text=label_override, fg=_DIM)
            return
        if not active_rules:
            self.rules_indicator.config(fg=COLORS['success'])
            self.rules_label.config(text="RULES: OK ✓", fg=COLORS['success'])
            return
        pri = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        top = min(active_rules, key=lambda r: pri.get(r['priority'], 4))
        p   = top['priority']
        n   = len(active_rules)
        if p == 'critical':
            c = COLORS['error'] if (int(time.time() * 3) % 2 == 0) else COLORS['error_light']
            self.rules_indicator.config(fg=c)
            self.rules_label.config(text=f"⚠ {n} CRITICAL", fg=c)
        elif p == 'high':
            self.rules_indicator.config(fg=COLORS['warning'])
            self.rules_label.config(text=f"⚠ {n} HIGH", fg=COLORS['warning'])
        else:
            self.rules_indicator.config(fg=COLORS['purple'])
            self.rules_label.config(text=f"ℹ {n} RULES", fg=COLORS['purple'])

    def update_driver_status(self, distracted=False, alerts=None, label_override=None):
        alerts = alerts or []
        if label_override:
            self.driver_indicator.config(fg=_DIM)
            self.driver_label.config(text=label_override, fg=_DIM)
        elif distracted and alerts:
            c = COLORS['error'] if (int(time.time() * 3) % 2 == 0) else COLORS['error_light']
            short = alerts[0][:18] + ("…" if len(alerts[0]) > 18 else "")
            self.driver_indicator.config(fg=c)
            self.driver_label.config(text=f"⚠ {short}", fg=c)
        elif distracted:
            self.driver_indicator.config(fg=COLORS['warning'])
            self.driver_label.config(text="⚠ DISTRACTED", fg=COLORS['warning'])
        else:
            self.driver_indicator.config(fg=COLORS['success'])
            self.driver_label.config(text="DRIVER: OK ✓", fg=COLORS['success'])