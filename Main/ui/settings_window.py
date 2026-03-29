"""Settings window for NeuroDrive — live apply on save"""
import tkinter as tk
from tkinter import ttk
from utils.constants import COLORS


class SettingsWindow:
    """Settings dialog — saves config AND applies changes live without restart."""

    def __init__(self, parent, config, alert_callback=None, apply_callback=None):
        self.parent          = parent
        self.config          = config
        self.alert_callback  = alert_callback
        self.apply_callback  = apply_callback   # ← called after save
        self.check_vars      = {}

        self.window = tk.Toplevel(parent)
        self.window.title("Settings - NeuroDrive")
        self.window.geometry("550x750")
        self.window.configure(bg=COLORS['bg_medium'])
        self.window.transient(parent)
        self.window.grab_set()

        self._create_ui()

    def _create_ui(self):
        # Header
        header = tk.Frame(self.window, bg=COLORS['bg_light'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)

        tk.Label(header, text="Alert Module Configuration",
                 font=("Helvetica", 18, "bold"),
                 fg=COLORS['accent'], bg=COLORS['bg_light']).pack(pady=15)
        tk.Label(header, text="Changes take effect immediately — no restart needed",
                 font=("Helvetica", 10),
                 fg=COLORS['text_tertiary'], bg=COLORS['bg_light']).pack()

        # Scrollable area
        canvas         = tk.Canvas(self.window, bg=COLORS['bg_medium'], highlightthickness=0)
        scrollbar      = ttk.Scrollbar(self.window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['bg_medium'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        style = ttk.Style()
        style.theme_use('default')
        style.configure("Vertical.TScrollbar",
                        background="#2a2a2a",
                        troughcolor=COLORS['bg_light'],
                        bordercolor=COLORS['bg_light'],
                        arrowcolor=COLORS['accent'])

        for module in self.config.config.keys():
            var = tk.BooleanVar(value=self.config.get(module))
            self.check_vars[module] = var

            mf = tk.Frame(scrollable_frame, bg=COLORS['bg_light'], relief='flat')
            mf.pack(fill='x', padx=20, pady=5)

            tk.Checkbutton(
                mf, text=module, variable=var,
                font=("Helvetica", 11),
                fg=COLORS['text_primary'], bg=COLORS['bg_light'],
                selectcolor="#2a2a2a",
                activebackground=COLORS['bg_light'],
                activeforeground=COLORS['accent'],
                cursor="hand2", relief='flat',
            ).pack(anchor='w', padx=15, pady=10)

        canvas.pack(side="left", fill="both", expand=True, padx=20, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)

        # Buttons
        bf = tk.Frame(self.window, bg=COLORS['bg_medium'])
        bf.pack(fill='x', pady=20)

        tk.Button(
            bf, text="Save & Apply",
            font=("Helvetica", 12, "bold"),
            bg=COLORS['accent'], fg="#000000",
            activebackground=COLORS['accent_dark'], activeforeground="#000000",
            relief='flat', padx=40, pady=12, cursor="hand2",
            command=self._save_settings,
        ).pack(side='left', padx=(40, 10))

        tk.Button(
            bf, text="Cancel",
            font=("Helvetica", 12),
            bg="#2a2a2a", fg=COLORS['text_secondary'],
            activebackground="#3a3a3a", activeforeground=COLORS['text_primary'],
            relief='flat', padx=40, pady=12, cursor="hand2",
            command=self.window.destroy,
        ).pack(side='left', padx=10)

    def _save_settings(self):
        for module, var in self.check_vars.items():
            self.config.set(module, var.get())

        saved = self.config.save()

        if saved:
            # Apply changes live
            if self.apply_callback:
                self.apply_callback()
            if self.alert_callback:
                self.alert_callback("✓ Configuration saved and applied")
        else:
            if self.alert_callback:
                self.alert_callback("❌ Error saving configuration")

        self.window.destroy()