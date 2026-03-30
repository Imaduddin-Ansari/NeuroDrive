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
from ui.summary_window import SummaryWindow


def main():
    """Main entry point for NeuroDrive application"""
    root = tk.Tk()
    app = NeuroDriveUI(root)
    
    def on_closing():
        """Handle window closing"""
        # Stop processors cleanly
        app.stop_video_feeds()

        # Build run summary (may be empty) and then destroy main UI
        try:
            summary = app.build_run_summary()
        except Exception:
            summary = {}

        root.destroy()

        # Open a focused summary window in a fresh Tk instance
        try:
            sum_root = tk.Tk()
            SummaryWindow(sum_root, summary)
            sum_root.mainloop()
        except Exception:
            # If GUI summary fails, fallback to printing a simple summary
            print("Session summary:")
            print(summary)
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()