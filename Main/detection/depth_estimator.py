"""Depth estimation module"""
import numpy as np


class AccurateDepthEstimator:
    """Estimates depth/distance to vehicles"""
    
    def __init__(self, focal_length=700, car_height=1.5, car_width=1.8):
        self.focal = focal_length
        self.car_h = car_height
        self.car_w = car_width

    def estimate_depth(self, bbox, shape):
        """
        Estimate depth/distance to a vehicle
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            shape: Frame shape (height, width, channels)
        
        Returns:
            Estimated depth in meters
        """
        x1, y1, x2, y2 = bbox
        h, w = shape[:2]
        bh, bw = y2 - y1, x2 - x1
        
        if bh <= 0:
            return 50.0
        
        # Depth from height
        d_h = self.focal * self.car_h / bh
        
        # Depth from width
        d_w = self.focal * self.car_w / bw if bw > 0 else d_h
        
        # Weighted average
        depth = 0.7 * d_h + 0.3 * d_w
        
        # Adjust based on vertical position
        vp = y2 / h
        if vp > 0.85:
            depth *= 0.7
        elif vp > 0.75:
            depth *= 0.85
        elif vp < 0.65:
            depth *= 1.2
        
        return float(np.clip(depth, 1.0, 100.0))