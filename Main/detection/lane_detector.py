"""Lane detection module"""
import cv2
import numpy as np


class RobustLaneDetector:
    """Perfect lane detection – always returns a lane (fallback if needed)"""
    
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.detection_confidence = 0.0
        self.smooth_alpha = 0.85
        self.frame_count = 0

    def preprocess_for_lanes(self, frame):
        h, w = frame.shape[:2]
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l = hls[:, :, 1]
        s = hls[:, :, 2]
        
        white = cv2.inRange(l, 180, 255)
        yellow = cv2.inRange(hls, (15, 30, 80), (35, 255, 255))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bin = cv2.inRange(gray, 200, 255)
        
        sobel = cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)
        sobel = np.uint8(255 * np.abs(sobel) / (np.max(np.abs(sobel)) + 1e-6))
        sobel_bin = cv2.inRange(sobel, 30, 255)
        
        combined = np.zeros_like(gray)
        combined[(white == 255) | (yellow == 255) | (gray_bin == 255) | (sobel_bin == 255)] = 255
        
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return combined

    def get_roi_mask(self, shape):
        h, w = shape[:2]
        vertices = np.array([
            [(int(w * 0.05), h), (int(w * 0.40), int(h * 0.55)),
             (int(w * 0.60), int(h * 0.55)), (int(w * 0.95), h)]
        ], dtype=np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, vertices, 255)
        return mask

    def detect_lane_lines_hough(self, frame):
        h, w = frame.shape[:2]
        binary = self.preprocess_for_lanes(frame)
        roi = cv2.bitwise_and(binary, self.get_roi_mask(frame.shape))
        edges = cv2.Canny(roi, 30, 90, apertureSize=3)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=150)
        if lines is None:
            return None, None, 0.0

        left, right = [], []
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x2 - x1) < 1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if not (0.3 <= abs(slope) <= 3.0):
                continue
            length = np.hypot(x2 - x1, y2 - y1)
            cx = (x1 + x2) / 2
            
            if slope < 0 and cx < w * 0.55:
                left.append((x1, y1, x2, y2, slope, length))
            elif slope > 0 and cx > w * 0.45:
                right.append((x1, y1, x2, y2, slope, length))

        left_lane = self._fit_lane(left, h, w, 'left')
        right_lane = self._fit_lane(right, h, w, 'right')
        conf = 1.0 if left_lane and right_lane else 0.6 if left_lane or right_lane else 0.0
        
        return left_lane, right_lane, conf

    def _fit_lane(self, lines, img_h, img_w, side):
        if len(lines) < 2:
            return None
        
        slopes = [l[4] for l in lines]
        lengths = [l[5] for l in lines]
        idx = np.argsort(slopes)
        slopes, lengths = [slopes[i] for i in idx], [lengths[i] for i in idx]
        
        cum = np.cumsum(lengths)
        median_slope = slopes[np.searchsorted(cum, cum[-1] / 2)]
        filtered = [l for l in lines if abs(l[4] - median_slope) < abs(median_slope * 0.4)]
        
        if len(filtered) < 2:
            filtered = lines
        
        xs, ys, ws = [], [], []
        for x1, y1, x2, y2, _, length in filtered:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
            ws.extend([length, length])
        
        try:
            coeffs = np.polyfit(ys, xs, 1, w=ws)
            y1, y2 = img_h, int(img_h * 0.55)
            x1, x2 = int(np.polyval(coeffs, y1)), int(np.polyval(coeffs, y2))
            
            if side == 'left':
                x1, x2 = np.clip(x1, 0, img_w // 2), np.clip(x2, 0, img_w // 2)
            else:
                x1, x2 = np.clip(x1, img_w // 2, img_w), np.clip(x2, img_w // 2, img_w)
            
            return {'coeffs': coeffs, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        except:
            return None

    def smooth_lanes(self, left, right):
        if self.left_fit and left:
            for k in ['coeffs', 'x1', 'x2']:
                left[k] = self.smooth_alpha * self.left_fit[k] + (1 - self.smooth_alpha) * left[k]
        if self.right_fit and right:
            for k in ['coeffs', 'x1', 'x2']:
                right[k] = self.smooth_alpha * self.right_fit[k] + (1 - self.smooth_alpha) * right[k]
        return left, right

    def fallback_lane_estimation(self, shape):
        h, w = shape[:2]
        return (
            {'coeffs': np.array([-0.7, w * 0.25]), 'x1': int(w * 0.20), 'y1': h,
             'x2': int(w * 0.40), 'y2': int(h * 0.60)},
            {'coeffs': np.array([0.7, w * 0.75]), 'x1': int(w * 0.80), 'y1': h,
             'x2': int(w * 0.60), 'y2': int(h * 0.60)}
        )

    def detect_lanes(self, frame):
        self.frame_count += 1
        left, right, conf = self.detect_lane_lines_hough(frame)
        
        if left or right:
            left, right = self.smooth_lanes(left, right)
        
        if left:
            self.left_fit = left
        if right:
            self.right_fit = right
        
        self.detection_confidence = conf if conf > 0 else self.detection_confidence * 0.9
        
        if self.detection_confidence < 0.3:
            if not self.left_fit or not self.right_fit:
                self.left_fit, self.right_fit = self.fallback_lane_estimation(frame.shape)
            self.detection_confidence = 0.5
        
        return self.left_fit, self.right_fit

    def is_in_ego_lane(self, bbox, shape):
        if not self.left_fit or not self.right_fit:
            h, w = shape[:2]
            cx = (bbox[0] + bbox[2]) / 2
            return w * 0.25 < cx < w * 0.75 and bbox[3] > h * 0.45
        
        h, w = shape[:2]
        cx, cy = (bbox[0] + bbox[2]) / 2, bbox[3]
        
        try:
            lx = np.polyval(self.left_fit['coeffs'], cy)
            rx = np.polyval(self.right_fit['coeffs'], cy)
        except:
            return False
        
        if rx <= lx:
            return False
        
        lane_w = rx - lx
        if not (w * 0.15 < lane_w < w * 0.7):
            return False
        
        margin = lane_w * 0.12
        return (lx + margin) < cx < (rx - margin)