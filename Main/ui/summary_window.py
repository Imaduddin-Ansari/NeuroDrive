import io
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SummaryWindow:
    """Display a visually appealing summary after an app run.

    The `summary` argument is expected to be a dict with optional keys:
      - 'duration_s' (float)
      - 'frames' (int)
      - 'pip' (dict) with metrics: 'frames_processed', 'avg_inference_time_ms',
        'fps', 'intent_histogram' (dict-like), 'alert_stats' (dict)
      - 'dsf' (dict) with 'raw_text' (str)
      - 'annotated_image' (PIL.Image or numpy.ndarray)

    The window shows charts on the left and a metrics table + text on the right.
    """

    def __init__(self, root, summary: dict):
        self.root = root
        self.root.title("NeuroDrive — Session Summary")
        self.summary = summary or {}

        self.root.geometry('1000x640')

        container = ttk.Frame(self.root, padding=10)
        container.pack(fill='both', expand=True)

        left = ttk.Frame(container)
        left.pack(side='left', fill='both', expand=True)

        right = ttk.Frame(container, width=360)
        right.pack(side='right', fill='y')

        # --- Matplotlib figure area (left top) ---
        fig = plt.Figure(figsize=(6, 3.6), dpi=100)
        self.axes = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
        fig.tight_layout(pad=2.0)

        canvas = FigureCanvasTkAgg(fig, master=left)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill='both', expand=False)

        # Draw charts from summary
        try:
            self._draw_charts(self.axes)
            canvas.draw()
        except Exception:
            # Keep UI robust even if plotting fails
            for ax in self.axes:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')

        # Thumbnail / annotated frame below charts
        thumb_frame = ttk.Frame(left, padding=(0, 8, 0, 0))
        thumb_frame.pack(fill='x', expand=False)
        self._thumb_label = ttk.Label(thumb_frame)
        self._thumb_label.pack()
        self._show_thumbnail()

        # --- Right pane: metrics table and raw DSF text ---
        title = ttk.Label(right, text='Run Metrics', font=('TkDefaultFont', 12, 'bold'))
        title.pack(anchor='nw', pady=(0, 6))

        cols = ('metric', 'value')
        self.tree = ttk.Treeview(right, columns=cols, show='headings', height=12)
        self.tree.heading('metric', text='Metric')
        self.tree.heading('value', text='Value')
        self.tree.column('metric', width=160, anchor='w')
        self.tree.column('value', width=180, anchor='center')
        self.tree.pack(fill='x')

        self._populate_table()

        # DSF text area
        dsf_label = ttk.Label(right, text='DSF Raw Summary', font=('TkDefaultFont', 11, 'bold'))
        dsf_label.pack(anchor='nw', pady=(8, 2))

        self.dsf_text = ScrolledText(right, wrap='word', height=12)
        self.dsf_text.pack(fill='both', expand=True)
        self._populate_dsf_text()

        # Close button
        btn = ttk.Button(right, text='Close', command=self.root.destroy)
        btn.pack(side='bottom', pady=(8, 0))

    # --- helpers ---
    def _draw_charts(self, axes):
        pip = self.summary.get('pip', {}) or {}

        # 1) Intent distribution (pie)
        intent_hist = pip.get('intent_histogram') or {}
        labels = list(intent_hist.keys()) if intent_hist else ['No data']
        sizes = list(intent_hist.values()) if intent_hist else [1]
        ax = axes[0]
        ax.clear()
        ax.set_title('Intent distribution')
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

        # 2) Alerts breakdown (bar)
        alert_stats = pip.get('alert_stats') or {}
        ax = axes[1]
        ax.clear()
        ax.set_title('Alerts')
        if alert_stats:
            keys = list(alert_stats.keys())
            vals = [alert_stats[k] for k in keys]
            ax.bar(keys, vals, color='orange')
            ax.set_ylabel('Count')
        else:
            ax.text(0.5, 0.5, 'No alerts', ha='center', va='center')

        # 3) Inference time box
        ax = axes[2]
        ax.clear()
        avg_ms = pip.get('avg_inference_time_ms')
        fps = pip.get('fps')
        ax.axis('off')
        ax.text(0.1, 0.6, f'Avg inference: {avg_ms:.1f} ms' if avg_ms else 'Avg inference: N/A', fontsize=10)
        ax.text(0.1, 0.3, f'FPS: {fps:.1f}' if fps else 'FPS: N/A', fontsize=10)

        # 4) Frames / duration
        ax = axes[3]
        ax.clear()
        frames = self.summary.get('frames')
        duration = self.summary.get('duration_s')
        ax.axis('off')
        ax.text(0.1, 0.6, f'Frames: {frames}' if frames else 'Frames: N/A', fontsize=10)
        ax.text(0.1, 0.3, f'Duration: {duration:.1f}s' if duration else 'Duration: N/A', fontsize=10)

    def _pil_from_maybe_ndarray(self, img):
        if img is None:
            return None
        try:
            if isinstance(img, Image.Image):
                return img
        except Exception:
            # PIL might not be imported as Image in this scope; continue
            pass

        # Assume numpy array (BGR or RGB)
        try:
            import numpy as _np
            from PIL import Image as _Image
            if _np and isinstance(img, _np.ndarray):
                arr = img.copy()
                # If BGR, convert to RGB heuristically by checking channel order
                if arr.ndim == 3 and arr.shape[2] == 3:
                    # detect likely BGR by mean of first channel vs last (not perfect)
                    if arr[..., 0].mean() > arr[..., 2].mean() + 1:
                        arr = arr[..., ::-1]
                return _Image.fromarray(arr)
        except Exception:
            pass
        return None

    def _show_thumbnail(self):
        img = self.summary.get('annotated_image')
        pil = self._pil_from_maybe_ndarray(img)
        if pil is None:
            # show placeholder
            self._thumb_label.config(text='No annotated image')
            return
        # resize to thumbnail width
        w = 360
        h = int(pil.height * (w / pil.width))
        thumb = pil.resize((w, h), Image.LANCZOS)
        tkimg = ImageTk.PhotoImage(thumb)
        # keep reference
        self._tk_thumbnail = tkimg
        self._thumb_label.config(image=tkimg)

    def _populate_table(self):
        # Clear
        for i in self.tree.get_children():
            self.tree.delete(i)

        # Basic run metrics
        self._add_metric('Duration (s)', self.summary.get('duration_s'))
        self._add_metric('Frames', self.summary.get('frames'))

        pip = self.summary.get('pip') or {}
        self._add_metric('PIP frames', pip.get('frames_processed'))
        self._add_metric('Avg inference (ms)', pip.get('avg_inference_time_ms'))
        self._add_metric('PIP FPS', pip.get('fps'))

        alert_stats = pip.get('alert_stats') or {}
        for k, v in alert_stats.items():
            self._add_metric(f'Alert: {k}', v)

    def _add_metric(self, name, value):
        if value is None:
            value = 'N/A'
        elif isinstance(value, float):
            value = f'{value:.2f}'
        self.tree.insert('', 'end', values=(name, value))

    def _populate_dsf_text(self):
        dsf = self.summary.get('dsf') or {}
        raw = dsf.get('raw_text') or dsf.get('summary_text') or ''
        if not raw:
            raw = 'No DSF text saved for this session.'
        self.dsf_text.insert('1.0', raw)
        self.dsf_text.configure(state='disabled')
