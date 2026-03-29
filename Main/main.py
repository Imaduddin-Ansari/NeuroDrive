"""
NeuroDrive - Advanced Driver Assistance System
Main entry point
"""
import tkinter as tk
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.main_window import NeuroDriveUI


def main():
    """Main entry point for NeuroDrive application"""
    root = tk.Tk()
    app = NeuroDriveUI(root)
    
    def on_closing():
        """Handle window closing"""
        app.stop_video_feeds()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()