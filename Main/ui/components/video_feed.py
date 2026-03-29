"""Video feed grid component — equal 2x2 grid, clean headers"""
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from utils.constants import COLORS, UI_DIMENSIONS

# Simple camera titles — no module names, no badges
FEED_TITLES = [
    "Left Side Camera",
    "Right Side Camera",
    "Front Camera",
    "Driver Camera",
]


class VideoFeedGrid(tk.Frame):
    """2×2 grid of equal-sized video feeds."""

    def __init__(self, parent, maximize_callback):
        super().__init__(parent, bg=COLORS['bg_medium'])
        self.maximize_callback = maximize_callback
        self.video_labels = []
        self._create_ui()

    def _create_ui(self):
        for i in range(4):
            row = i // 2
            col = i % 2

            # Thin accent border
            outer = tk.Frame(self, bg=COLORS['accent'], relief='flat')
            outer.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')

            inner = tk.Frame(outer, bg=COLORS['bg_light'], relief='flat')
            inner.pack(fill='both', expand=True, padx=2, pady=2)

            # Header — fixed height, simple title only
            header = tk.Frame(
                inner,
                bg=COLORS['bg_lighter'],
                height=UI_DIMENSIONS['feed_header_height'],
            )
            header.pack(fill='x', side='top')
            header.pack_propagate(False)

            tk.Label(
                header,
                text=FEED_TITLES[i],
                font=("Helvetica", 10, "bold"),
                fg=COLORS['accent'],
                bg=COLORS['bg_lighter'],
            ).pack(side='left', padx=10, pady=5)

            # Video label
            content = tk.Frame(inner, bg=COLORS['bg_light'])
            content.pack(fill='both', expand=True)

            lbl = tk.Label(
                content,
                text="📹",
                font=("Helvetica", 24),
                fg=COLORS['text_dim'],
                bg=COLORS['bg_light'],
                cursor="hand2",
            )
            lbl.pack(expand=True, fill='both')
            lbl.bind("<Button-1>", lambda e, idx=i: self.maximize_callback(idx))

            self.video_labels.append(lbl)

        # Equal weights + uniform group so both columns are always identical width
        for r in range(2):
            self.grid_rowconfigure(r, weight=1, uniform='row')
        for c in range(2):
            self.grid_columnconfigure(c, weight=1, uniform='col')

    def update_feed(self, feed_index, frame):
        """Replace the placeholder with a live video frame."""
        if feed_index >= len(self.video_labels):
            return

        lbl = self.video_labels[feed_index]

        # BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        w = lbl.winfo_width()
        h = lbl.winfo_height()
        if w <= 1 or h <= 1:
            w, h = 400, 280

        fh, fw = frame.shape[:2]
        aspect = fw / fh

        if w / h > aspect:
            new_h = h
            new_w = int(aspect * new_h)
        else:
            new_w = w
            new_h = int(new_w / aspect)

        frame = cv2.resize(frame, (new_w, new_h))
        img   = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        if hasattr(lbl, 'imgtk'):
            del lbl.imgtk

        lbl.imgtk = imgtk
        lbl.configure(image=imgtk, text="")