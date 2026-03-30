"""
Alert panel — split bottom bar:
  LEFT  (60%): LLM Risk Explanations + Priority Rule Alerts
  RIGHT (40%): Driving Style Feedback live stats
"""
import tkinter as tk
from utils.constants import COLORS, ALERT_SETTINGS, UI_DIMENSIONS


class AlertPanel(tk.Frame):
    """Bottom bar split: left = alerts/LLM, right = DSF live stats."""

    def __init__(self, parent):
        super().__init__(parent, bg=COLORS['bg_light'],
                         height=UI_DIMENSIONS['alert_panel_height'])
        self.pack_propagate(False)
        self.displayed_rules = {}
        self.rule_cooldown   = 20.0
        self._last_llm_text  = ""
        self._last_llm_time  = 0.0
        self._llm_cooldown   = 15.0
        self.current_alerts  = []
        self._build()

    def _build(self):
        outer = tk.Frame(self, bg=COLORS['bg_light'])
        outer.pack(fill='both', expand=True)
        outer.grid_columnconfigure(0, weight=6)
        outer.grid_columnconfigure(1, weight=4)
        outer.grid_rowconfigure(0, weight=1)
        self._build_left(outer)
        tk.Frame(outer, bg=COLORS['bg_lighter'], width=2).grid(
            row=0, column=1, sticky='ns', pady=4)
        self._build_right(outer)

    def _build_left(self, parent):
        f = tk.Frame(parent, bg=COLORS['bg_light'])
        f.grid(row=0, column=0, sticky='nsew', padx=(5, 2), pady=5)
        hdr = tk.Frame(f, bg=COLORS['bg_lighter'],
                       height=UI_DIMENSIONS['feed_header_height'])
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text="\U0001f514  ALERTS & RISK EXPLANATIONS",
                 font=("Helvetica", 11, "bold"),
                 fg=COLORS['accent'], bg=COLORS['bg_lighter']
                 ).pack(anchor='w', padx=12, pady=6)
        self.alert_text = tk.Text(
            f, font=("Helvetica", 13, "bold"),
            fg=COLORS['text_primary'], bg=COLORS['bg_light'],
            height=1, wrap='word', relief='flat', padx=12, pady=8)
        self.alert_text.pack(fill='both', expand=True)
        self.alert_text.insert('1.0', "\u2713 System Active \u2014 Monitoring\u2026")
        self.alert_text.config(state='disabled')
        self.alert_text.tag_config("critical", foreground=COLORS['error'])
        self.alert_text.tag_config("high",     foreground=COLORS['warning'])
        self.alert_text.tag_config("medium",   foreground=COLORS['purple_light'])
        self.alert_text.tag_config("low",      foreground=COLORS['text_secondary'])
        self.alert_text.tag_config("llm",
            foreground=COLORS['accent'],
            font=("Helvetica", 12, "italic"))

    def _build_right(self, parent):
        f = tk.Frame(parent, bg=COLORS['bg_light'])
        f.grid(row=0, column=1, sticky='nsew', padx=(2, 5), pady=5)
        hdr = tk.Frame(f, bg=COLORS['bg_lighter'],
                       height=UI_DIMENSIONS['feed_header_height'])
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text="\U0001f697  DRIVING STYLE FEEDBACK",
                 font=("Helvetica", 11, "bold"),
                 fg='#00ff66', bg=COLORS['bg_lighter']
                 ).pack(anchor='w', padx=12, pady=6)
        body = tk.Frame(f, bg=COLORS['bg_light'])
        body.pack(fill='both', expand=True, padx=8, pady=4)
        row0 = tk.Frame(body, bg=COLORS['bg_light'])
        row0.pack(fill='x')
        self._score_var = tk.StringVar(value="100")
        self._tier_var  = tk.StringVar(value="Safe")
        sf = tk.Frame(row0, bg=COLORS['bg_light'])
        sf.pack(side='left')
        tk.Label(sf, text="SCORE", font=("Helvetica", 8),
                 fg=COLORS['text_dim'], bg=COLORS['bg_light']).pack()
        self._score_lbl = tk.Label(sf, textvariable=self._score_var,
                                   font=("Helvetica", 26, "bold"),
                                   fg='#00ff66', bg=COLORS['bg_light'])
        self._score_lbl.pack()
        tf = tk.Frame(row0, bg=COLORS['bg_light'])
        tf.pack(side='left', padx=10)
        tk.Label(tf, text="TIER", font=("Helvetica", 8),
                 fg=COLORS['text_dim'], bg=COLORS['bg_light']).pack()
        self._tier_lbl = tk.Label(tf, textvariable=self._tier_var,
                                  font=("Helvetica", 14, "bold"),
                                  fg='#00ff66', bg=COLORS['bg_light'])
        self._tier_lbl.pack()
        row1 = tk.Frame(body, bg=COLORS['bg_light'])
        row1.pack(fill='x', pady=2)
        self._elapsed_var = tk.StringVar(value="00:00")
        self._dist_var    = tk.StringVar(value="0.000 km")
        self._rw_var      = tk.StringVar(value="1.00x")
        for lbl, var in [("ELAPSED", self._elapsed_var),
                          ("DISTANCE", self._dist_var),
                          ("RISK WT",  self._rw_var)]:
            c = tk.Frame(row1, bg=COLORS['bg_light'])
            c.pack(side='left', padx=6)
            tk.Label(c, text=lbl, font=("Helvetica", 7),
                     fg=COLORS['text_dim'], bg=COLORS['bg_light']).pack()
            tk.Label(c, textvariable=var, font=("Helvetica", 10, "bold"),
                     fg=COLORS['text_secondary'], bg=COLORS['bg_light']).pack()
        row2 = tk.Frame(body, bg=COLORS['bg_light'])
        row2.pack(fill='x', pady=2)
        self._accel_cnt = tk.StringVar(value="0")
        self._brake_cnt = tk.StringVar(value="0")
        self._steer_cnt = tk.StringVar(value="0")
        for lbl, var, col in [("HARSH ACCEL", self._accel_cnt, COLORS['warning']),
                               ("HARSH BRAKE", self._brake_cnt, COLORS['error']),
                               ("AGGR STEER",  self._steer_cnt, COLORS['purple_light'])]:
            c = tk.Frame(row2, bg=COLORS['bg_light'])
            c.pack(side='left', padx=6)
            tk.Label(c, text=lbl, font=("Helvetica", 7),
                     fg=COLORS['text_dim'], bg=COLORS['bg_light']).pack()
            tk.Label(c, textvariable=var, font=("Helvetica", 12, "bold"),
                     fg=col, bg=COLORS['bg_light']).pack()
        self._alerts_var = tk.StringVar(value="")
        tk.Label(body, textvariable=self._alerts_var,
                 font=("Helvetica", 9), fg=COLORS['warning'],
                 bg=COLORS['bg_light'], wraplength=320, justify='left'
                 ).pack(fill='x', pady=(2, 0))

    _TIER_COLORS = {"Safe": "#00ff66", "Moderate": "#ffaa00", "Aggressive": "#ff0000"}

    def update_dsf(self, data):
        score = data.get("score", 100)
        tier  = data.get("tier",  "Safe")
        color = self._TIER_COLORS.get(tier, "#00ff66")
        self._score_var.set(str(score))
        self._tier_var.set(tier)
        self._score_lbl.config(fg=color)
        self._tier_lbl.config(fg=color)
        elapsed = data.get("elapsed_s", 0)
        self._elapsed_var.set(f"{int(elapsed//60):02d}:{int(elapsed%60):02d}")
        self._dist_var.set(f"{data.get('distance_km', 0):.3f} km")
        self._rw_var.set(f"{data.get('risk_weight', 1.0):.2f}x")
        ec = data.get("event_counts", {})
        self._accel_cnt.set(str(ec.get("harsh_accel",      0)))
        self._brake_cnt.set(str(ec.get("harsh_brake",       0)))
        self._steer_cnt.set(str(ec.get("aggressive_steer", 0)))
        active = data.get("active_alerts", {})
        parts = []
        if active.get("fcw_alert"):         parts.append("\u26a0 FCW")
        if active.get("lane_alert"):        parts.append("\u26a0 LANE")
        if active.get("bsp_alert"):         parts.append("\u26a0 BSP")
        if active.get("sign_alert"):        parts.append("\u26a0 SIGN")
        if active.get("priority_alert"):    parts.append("\u26a0 RULES")
        if active.get("distraction_alert"): parts.append("\u26a0 DRIVER")
        if active.get("pip_alert"):         parts.append("\u26a0 PED")
        self._alerts_var.set("  ".join(parts) if parts else "")

    def add_priority_rule(self, rule_dict, current_time):
        desc = rule_dict.get('description', 'Unknown rule')
        if desc in self.displayed_rules:
            if current_time - self.displayed_rules[desc] < self.rule_cooldown:
                return
        self.displayed_rules[desc] = current_time
        p = rule_dict.get('priority', 'unknown').upper()
        if p == 'CRITICAL':   emoji, tag = "\U0001f6a8", "critical"
        elif p == 'HIGH':     emoji, tag = "\u26a0\ufe0f", "high"
        elif p == 'MEDIUM':   emoji, tag = "\u2139\ufe0f", "medium"
        else:                  emoji, tag = "\U0001f4cb", "low"
        self._append_line(f"{emoji} {desc}", tag)
        self._cleanup_old_rules(current_time)

    def add_llm_explanation(self, text, current_time):
        if not text:
            return
        if (text == self._last_llm_text and
                current_time - self._last_llm_time < self._llm_cooldown):
            return
        self._last_llm_text = text
        self._last_llm_time = current_time
        self._append_line(f"\U0001f916 {text}", "llm")

    def _append_line(self, message, tag=""):
        self.alert_text.config(state='normal')
        current = self.alert_text.get('1.0', 'end-1c')
        lines = [l for l in current.split('\n') if l.strip()] if current.strip() else []
        max_lines = ALERT_SETTINGS['max_lines']
        if len(lines) >= max_lines:
            lines = lines[-(max_lines - 1):]
        self.alert_text.delete('1.0', 'end')
        for line in lines:
            self.alert_text.insert('end', line + '\n', self._tag_for_line(line))
        self.alert_text.insert('end', message + '\n', tag)
        self.alert_text.config(state='disabled')
        self.alert_text.see('end')

    @staticmethod
    def _tag_for_line(line):
        if "\U0001f6a8" in line: return "critical"
        if "\u26a0\ufe0f" in line: return "high"
        if "\u2139\ufe0f" in line: return "medium"
        if "\U0001f916" in line: return "llm"
        return ""

    def _cleanup_old_rules(self, current_time):
        stale = [k for k, ts in self.displayed_rules.items()
                 if current_time - ts > self.rule_cooldown * 2]
        for k in stale:
            del self.displayed_rules[k]

    def add_message(self, message):
        pass

    def has_message(self, text):
        return text in self.alert_text.get('1.0', 'end-1c')

    def clear(self):
        self.alert_text.config(state='normal')
        self.alert_text.delete('1.0', 'end')
        self.alert_text.insert('1.0', "\u2713 System Active \u2014 Monitoring\u2026")
        self.alert_text.config(state='disabled')
        self.displayed_rules.clear()
        self.current_alerts.clear()
        self._last_llm_text = ""
