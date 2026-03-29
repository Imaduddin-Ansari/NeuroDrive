"""Base processor class for all video processors"""
import threading
import queue
import time


class BaseProcessor:
    """Base class for all video processing modules"""
    
    def __init__(self, video_source):
        self.video_source = video_source
        self.is_running = False
        self.thread = None
        # Small queue — we only ever need the NEWEST frame.
        # A large queue causes stale-frame backlog (alerts fire before video).
        self.frame_queue = queue.Queue(maxsize=2)
        self.detection_status = False
    
    def start(self):
        """Start processing in a separate thread"""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop processing"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _run(self):
        """Main processing loop - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _run()")
    
    def get_processed_frame(self):
        """Get the latest processed frame"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _put_frame(self, frame):
        """Put a frame in the queue — drop oldest if full to stay live"""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop oldest frame and put new one
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
    
    def _pace_to_fps(self, fps, last_frame_time):
        """
        Pace the processing loop to match the video's native FPS.
        Returns the updated last_frame_time.
        Call this after reading each frame.
        """
        if fps and fps > 0:
            target_interval = 1.0 / fps
            elapsed = time.time() - last_frame_time
            sleep_time = target_interval - elapsed
            if sleep_time > 0.002:
                time.sleep(sleep_time)
        return time.time()
    
    def _sleep(self, duration=0.01):
        """Sleep for a specified duration — default reduced for better FPS"""
        time.sleep(duration)