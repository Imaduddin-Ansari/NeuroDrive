"""
calibration_overlay.py
Handles the driver calibration gate — shows live webcam feed
until driver_profile.json is written, then fires a callback.
Extracted from main_window.py to keep it lean.
"""

import tkinter as tk
import cv2
import os
from pathlib import Path
from PIL import Image, ImageTk

from utils.constants import COLORS


class CalibrationOverlay:
    """
    Floating window that streams the live driver-camera feed during
    first-time calibration.  Fires `on_complete` callback when done.
    """

    _FRAME_MS = 30   # ~33 fps refresh for the calibration preview

    def __init__(self, parent, driver_processor, on_complete):
        self.parent           = parent
        self.driver_processor = driver_processor
        self.on_complete      = on_complete
        self._after_id        = None
        self._win             = None

    # ──────────────────────────────────────────────────────────────────────
    def show(self):
        """Open the calibration window and start polling."""
        self._win = tk.Toplevel(self.parent)
        self._win.title("NeuroDrive — Driver Calibration")
        self._win.configure(bg=COLORS['bg_dark'])
        self._win.geometry("960x620")
        self._win.resizable(True, True)

        # Header
        hdr = tk.Frame(self._win, bg=COLORS['bg_light'], height=52)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(
            hdr, text="NeuroDrive  —  Driver Calibration",
            font=("Helvetica", 14, "bold"),
            fg=COLORS['accent'], bg=COLORS['bg_light'],
        ).pack(side='left', padx=16, pady=12)

        # Separator
        tk.Frame(self._win, bg=COLORS['accent'], height=2).pack(fill='x')

        # Live video label
        self._video_label = tk.Label(self._win, bg='black')
        self._video_label.pack(expand=True, fill='both', padx=10, pady=10)

        # Footer
        tk.Label(
            self._win,
            text="Follow the on-screen instructions.  "
                 "This window closes automatically when done.",
            font=("Helvetica", 10),
            fg=COLORS['text_tertiary'], bg=COLORS['bg_dark'],
        ).pack(pady=8)

        self._update_frame()
        self._poll_done()

    # ──────────────────────────────────────────────────────────────────────
    def _update_frame(self):
        """Refresh the live video frame inside the calibration window."""
        if self._win is None or not self._win.winfo_exists():
            return

        if self.driver_processor and self.driver_processor.is_running:
            frame = self.driver_processor.get_processed_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                w = max(1, self._video_label.winfo_width())
                h = max(1, self._video_label.winfo_height())
                fh, fw = frame_rgb.shape[:2]
                scale  = min(w / fw, h / fh)
                nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
                resized = cv2.resize(frame_rgb, (nw, nh))
                imgtk  = ImageTk.PhotoImage(image=Image.fromarray(resized))
                self._video_label.imgtk = imgtk
                self._video_label.configure(image=imgtk)

        if not self.driver_processor.calibration_complete.is_set():
            self._win.after(self._FRAME_MS, self._update_frame)

    # ──────────────────────────────────────────────────────────────────────
    def _poll_done(self):
        """Check every 100 ms whether calibration is complete."""
        if self.driver_processor is None:
            self._finish()
            return

        if self.driver_processor.calibration_complete.is_set():
            self._finish()
        else:
            self._after_id = self._win.after(100, self._poll_done)

    def _finish(self):
        """Close the overlay and fire the completion callback."""
        if self._win is not None and self._win.winfo_exists():
            self._win.destroy()
        self._win = None
        self.on_complete()


# ──────────────────────────────────────────────────────────────────────────────
class NeuroDrivePopup:
    """
    Dark-themed modal dialog for startup warnings / errors.
    Replaces native OS alert sheets with a NeuroDrive-styled popup.
    """

    def __init__(self, parent, title, messages, level='warning'):
        accent = COLORS['warning'] if level == 'warning' else COLORS['error']

        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.configure(bg=COLORS['bg_medium'])
        self.win.resizable(False, False)
        self.win.transient(parent)
        self.win.grab_set()

        # Header
        header = tk.Frame(self.win, bg=COLORS['bg_light'], height=64)
        header.pack(fill='x')
        header.pack_propagate(False)

        icon = "⚠" if level == 'warning' else "✕"
        tk.Label(header, text=icon,
                 font=("Helvetica", 26, "bold"),
                 fg=accent, bg=COLORS['bg_light']).pack(side='left', padx=18, pady=12)
        tk.Label(header, text=title,
                 font=("Helvetica", 15, "bold"),
                 fg=COLORS['text_primary'], bg=COLORS['bg_light']).pack(side='left')

        # Separator
        tk.Frame(self.win, bg=accent, height=2).pack(fill='x')

        # Body
        body = tk.Frame(self.win, bg=COLORS['bg_medium'])
        body.pack(fill='both', expand=True, padx=24, pady=16)
        tk.Label(
            body,
            text="The following issues were encountered during startup.\n"
                 "The system will continue with reduced functionality where affected.",
            font=("Helvetica", 11),
            fg=COLORS['text_secondary'], bg=COLORS['bg_medium'],
            justify='left', wraplength=460,
        ).pack(anchor='w', pady=(0, 12))

        for msg in messages:
            row = tk.Frame(body, bg=COLORS['bg_medium'])
            row.pack(fill='x', pady=3)
            tk.Label(row, text="•", font=("Helvetica", 13, "bold"),
                     fg=accent, bg=COLORS['bg_medium']).pack(side='left', padx=(0, 8))
            tk.Label(row, text=msg, font=("Helvetica", 10),
                     fg=COLORS['text_secondary'], bg=COLORS['bg_medium'],
                     justify='left', wraplength=420).pack(side='left', anchor='w')

        # Button
        btn_frame = tk.Frame(self.win, bg=COLORS['bg_medium'])
        btn_frame.pack(pady=(4, 20))
        tk.Button(
            btn_frame, text="OK  —  Continue to NeuroDrive",
            font=("Helvetica", 11, "bold"),
            bg=accent, fg="#000000",
            activebackground=COLORS['bg_lighter'],
            activeforeground=COLORS['text_primary'],
            relief='flat', padx=28, pady=10, cursor="hand2",
            command=self.win.destroy,
        ).pack()

        # Centre on parent
        self.win.update_idletasks()
        pw = parent.winfo_width();  ph = parent.winfo_height()
        px = parent.winfo_rootx(); py = parent.winfo_rooty()
        ww = self.win.winfo_width(); wh = self.win.winfo_height()
        self.win.geometry(f"+{px + (pw - ww)//2}+{py + (ph - wh)//2}")

        parent.wait_window(self.win)