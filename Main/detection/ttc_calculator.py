"""Time-to-collision calculation module"""
import numpy as np
from collections import deque


class TTCCalculator:
    """Calculates time-to-collision for tracked vehicles"""
    
    def __init__(self, history_length=5):
        self.history = {}
        self.len = history_length

    def update(self, track_id, depth, timestamp):
        """
        Update tracking history for a vehicle
        
        Args:
            track_id: Unique ID of the tracked vehicle
            depth: Current depth/distance
            timestamp: Current time
        """
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.len)
        self.history[track_id].append({'d': depth, 't': timestamp})

    def calculate_ttc(self, track_id, ego_speed_mps):
        """
        Calculate time-to-collision
        
        Args:
            track_id: Unique ID of the tracked vehicle
            ego_speed_mps: Ego vehicle speed in m/s
        
        Returns:
            Tuple of (ttc, relative_velocity)
            ttc: Time to collision in seconds (inf if no collision)
            relative_velocity: Relative velocity in m/s
        """
        if track_id not in self.history or len(self.history[track_id]) < 3:
            return float('inf'), 0.0
        
        hist = list(self.history[track_id])[-3:]
        ds = [p['d'] for p in hist]
        ts = [p['t'] for p in hist]
        
        try:
            dt = np.diff(ts)
            dd = np.diff(ds)
            
            if np.sum(dt) == 0:
                return float('inf'), 0.0
            
            # Relative velocity (negative = approaching)
            rel_v = np.mean(dd / dt)
            
            # Closing speed
            closing = ego_speed_mps - rel_v
            
            if closing <= 0.5:
                return float('inf'), rel_v
            
            # Time to collision
            ttc = ds[-1] / closing
            
            return ttc if 0 < ttc <= 20 else float('inf'), rel_v
        except:
            return float('inf'), 0.0
    
    # Alias for backward compatibility
    def ttc(self, track_id, ego_speed_mps):
        return self.calculate_ttc(track_id, ego_speed_mps)