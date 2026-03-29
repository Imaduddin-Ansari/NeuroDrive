"""Loading screen for NeuroDrive"""
import tkinter as tk
from utils.constants import COLORS


class LoadingScreen:
    """Loading screen with NeuroDrive branding"""
    
    def __init__(self, parent):
        self.parent = parent
        self.frame = None
        self.loading_label = None
    
    def show(self):
        """Display loading screen"""
        self.frame = tk.Frame(self.parent, bg=COLORS['bg_dark'])
        self.frame.pack(fill='both', expand=True)
        
        title_label = tk.Label(
            self.frame,
            text="NeuroDrive",
            font=("Helvetica", 56, "bold"),
            fg=COLORS['accent'],
            bg=COLORS['bg_dark']
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            self.frame,
            text="Advanced Driver Assistance System",
            font=("Helvetica", 14),
            fg=COLORS['text_tertiary'],
            bg=COLORS['bg_dark']
        )
        subtitle_label.pack()
        
        self.loading_label = tk.Label(
            self.frame,
            text="Initializing System...",
            font=("Helvetica", 12),
            fg=COLORS['accent'],
            bg=COLORS['bg_dark']
        )
        self.loading_label.pack(pady=20)
        
        progress_frame = tk.Frame(self.frame, bg=COLORS['bg_light'], height=4, width=300)
        progress_frame.pack()
        progress_fill = tk.Frame(progress_frame, bg=COLORS['accent'], height=4)
        progress_fill.pack(side='left', fill='y')
        
        def animate_progress(width=0):
            if width <= 300 and self.frame.winfo_exists():
                progress_fill.config(width=width)
                self.parent.after(15, lambda: animate_progress(width + 5))
        
        animate_progress()
    
    def update_message(self, message):
        """Update loading message"""
        if self.loading_label and self.loading_label.winfo_exists():
            self.loading_label.config(text=message)
    
    def hide(self):
        """Hide loading screen"""
        if self.frame and self.frame.winfo_exists():
            self.frame.destroy()