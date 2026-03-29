"""Alert panel component — Priority Rules + LLM Risk Explanations"""
import tkinter as tk
from utils.constants import COLORS, ALERT_SETTINGS, UI_DIMENSIONS
from utils.helpers import format_timestamp


class AlertPanel(tk.Frame):
    """
    Alert and notification panel.

    Shows two categories of message:
      • Priority-rule alerts  (from PriorityRulesProcessor)
      • LLM risk explanations (from LLMRiskExplainer)

    LLM explanations are displayed in the accent colour so the driver
    can distinguish a sensor label ("🚨 Stop sign …") from an
    actionable sentence ("Brake gradually — car 8 m ahead while a
    cyclist is passing on your left.").
    """

    def __init__(self, parent):
        super().__init__(
            parent,
            bg=COLORS['bg_light'],
            height=UI_DIMENSIONS['alert_panel_height'],
        )
        self.pack_propagate(False)

        # Spam-prevention tracking for priority rules
        self.displayed_rules = {}   # description -> last-shown timestamp
        self.rule_cooldown   = 20.0

        # LLM explanation de-dup (same text within 15 s → skip)
        self._last_llm_text = ""
        self._last_llm_time = 0.0
        self._llm_cooldown  = 15.0

        self.current_alerts: list = []

        self._create_ui()

    # ─────────────────────────────────────────────────────────────────────────

    def _create_ui(self):
        """Build the panel: header strip + scrollable text area."""
        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(
            self,
            bg=COLORS['bg_lighter'],
            height=UI_DIMENSIONS['feed_header_height'],
        )
        header.pack(fill='x', side='top')
        header.pack_propagate(False)

        tk.Label(
            header,
            text="🔔  ALERTS & RISK EXPLANATIONS",
            font=("Helvetica", 12, "bold"),
            fg=COLORS['accent'],
            bg=COLORS['bg_lighter'],
        ).pack(anchor='w', padx=15, pady=8)

        # ── Text area ─────────────────────────────────────────────────────────
        self.alert_text = tk.Text(
            self,
            font=("Helvetica", 14, "bold"),
            fg=COLORS['text_primary'],
            bg=COLORS['bg_light'],
            height=1,
            wrap='word',
            relief='flat',
            padx=15,
            pady=10,
        )
        self.alert_text.pack(fill='both', expand=True, padx=5, pady=5)
        self.alert_text.insert('1.0', "✓ System Active — Monitoring…")
        self.alert_text.config(state='disabled')

        # Colour tags
        self.alert_text.tag_config("critical", foreground=COLORS['error'])
        self.alert_text.tag_config("high",     foreground=COLORS['warning'])
        self.alert_text.tag_config("medium",   foreground=COLORS['purple_light'])
        self.alert_text.tag_config("low",      foreground=COLORS['text_secondary'])
        self.alert_text.tag_config(
            "llm",
            foreground=COLORS['accent'],
            font=("Helvetica", 13, "italic"),
        )

    # ── Priority rule alerts ──────────────────────────────────────────────────

    def add_priority_rule(self, rule_dict: dict, current_time: float) -> None:
        """
        Add a priority-rule alert with spam prevention.
        The same description is suppressed for rule_cooldown seconds.
        """
        description = rule_dict.get('description', 'Unknown rule')

        if description in self.displayed_rules:
            if current_time - self.displayed_rules[description] < self.rule_cooldown:
                return

        self.displayed_rules[description] = current_time

        priority = rule_dict.get('priority', 'unknown').upper()
        if priority == 'CRITICAL':
            emoji, tag = "🚨", "critical"
        elif priority == 'HIGH':
            emoji, tag = "⚠️", "high"
        elif priority == 'MEDIUM':
            emoji, tag = "ℹ️", "medium"
        else:
            emoji, tag = "📋", "low"

        self._append_line(f"{emoji} {description}", tag)
        self._cleanup_old_rules(current_time)

    # ── LLM explanations ──────────────────────────────────────────────────────

    def add_llm_explanation(self, text: str, current_time: float) -> None:
        """
        Display an LLM-generated explanation in accent/italic style.

        Identical text within _llm_cooldown seconds is suppressed so a
        repeated slow alert doesn't spam the panel.
        """
        if not text:
            return
        if (text == self._last_llm_text
                and current_time - self._last_llm_time < self._llm_cooldown):
            return

        self._last_llm_text = text
        self._last_llm_time = current_time
        self._append_line(f"🤖 {text}", "llm")

    # ── Generic helpers ───────────────────────────────────────────────────────

    def _append_line(self, message: str, tag: str = "") -> None:
        """Append one line to the text widget, trim to max_lines."""
        self.alert_text.config(state='normal')

        current = self.alert_text.get('1.0', 'end-1c')
        lines   = [l for l in current.split('\n') if l.strip()] if current.strip() else []

        max_lines = ALERT_SETTINGS['max_lines']
        if len(lines) >= max_lines:
            lines = lines[-(max_lines - 1):]

        # Rebuild — reinsert existing lines then append new one
        self.alert_text.delete('1.0', 'end')
        for line in lines:
            existing_tag = self._tag_for_line(line)
            self.alert_text.insert('end', line + '\n', existing_tag)

        self.alert_text.insert('end', message + '\n', tag)
        self.alert_text.config(state='disabled')
        self.alert_text.see('end')

    @staticmethod
    def _tag_for_line(line: str) -> str:
        """Re-derive the colour tag for a line when rebuilding the widget."""
        if "🚨" in line:      return "critical"
        if "⚠️" in line:     return "high"
        if "ℹ️" in line:     return "medium"
        if "🤖" in line:      return "llm"
        return ""

    def _cleanup_old_rules(self, current_time: float) -> None:
        stale = [
            k for k, ts in self.displayed_rules.items()
            if current_time - ts > self.rule_cooldown * 2
        ]
        for k in stale:
            del self.displayed_rules[k]

    # ── Back-compat stubs ─────────────────────────────────────────────────────

    def add_message(self, message: str) -> None:
        """Legacy no-op kept for backward compatibility."""
        pass

    def has_message(self, text: str) -> bool:
        current = self.alert_text.get('1.0', 'end-1c')
        return text in current

    def clear(self) -> None:
        self.alert_text.config(state='normal')
        self.alert_text.delete('1.0', 'end')
        self.alert_text.insert('1.0', "✓ System Active — Monitoring…")
        self.alert_text.config(state='disabled')
        self.displayed_rules.clear()
        self.current_alerts.clear()
        self._last_llm_text = ""