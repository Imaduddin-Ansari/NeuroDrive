"""Helper utility functions for NeuroDrive"""
import time
import cv2
import numpy as np
from PIL import Image, ImageTk


def format_timestamp():
    """Format current time as HH:MM:SS"""
    return time.strftime('%H:%M:%S')


def resize_frame_to_fit(frame, max_width, max_height):
    """
    Resize frame to fit within max dimensions while maintaining aspect ratio
    
    Args:
        frame: numpy array (BGR image)
        max_width: maximum width
        max_height: maximum height
    
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    aspect = w / h
    
    if max_width / max_height > aspect:
        new_h = max_height
        new_w = int(aspect * new_h)
    else:
        new_w = max_width
        new_h = int(new_w / aspect)
    
    return cv2.resize(frame, (new_w, new_h))


def frame_to_photoimage(frame):
    """
    Convert OpenCV frame to Tkinter PhotoImage
    
    Args:
        frame: numpy array (BGR image)
    
    Returns:
        ImageTk.PhotoImage object
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    return ImageTk.PhotoImage(image=img)


def calculate_fps(frame_times, window=30):
    """
    Calculate average FPS from frame timestamps
    
    Args:
        frame_times: list of timestamps
        window: number of frames to average over
    
    Returns:
        Average FPS
    """
    if len(frame_times) < 2:
        return 0.0
    
    recent_times = frame_times[-window:]
    if len(recent_times) < 2:
        return 0.0
    
    time_diff = recent_times[-1] - recent_times[0]
    if time_diff == 0:
        return 0.0
    
    return (len(recent_times) - 1) / time_diff


def draw_box(frame, bbox, color=(0, 255, 0), thickness=2, label=None):
    """
    Draw bounding box on frame
    
    Args:
        frame: numpy array (BGR image)
        bbox: tuple (x1, y1, x2, y2)
        color: BGR color tuple
        thickness: line thickness
        label: optional text label
    
    Returns:
        Frame with box drawn
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Draw text background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return frame


def draw_text_with_background(frame, text, position, font_scale=1.0, 
                              color=(255, 255, 255), bg_color=(0, 0, 0),
                              thickness=2, padding=10):
    """
    Draw text with background rectangle
    
    Args:
        frame: numpy array (BGR image)
        text: text to draw
        position: (x, y) position
        font_scale: font scale
        color: text color (BGR)
        bg_color: background color (BGR)
        thickness: text thickness
        padding: padding around text
    
    Returns:
        Frame with text drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position
    
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Draw background
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness
    )
    
    return frame


def interpolate_color(value, min_val, max_val, color_low=(0, 255, 0), color_high=(0, 0, 255)):
    """
    Interpolate between two colors based on value
    
    Args:
        value: current value
        min_val: minimum value (maps to color_low)
        max_val: maximum value (maps to color_high)
        color_low: BGR color for min value
        color_high: BGR color for max value
    
    Returns:
        Interpolated BGR color
    """
    if value <= min_val:
        return color_low
    if value >= max_val:
        return color_high
    
    ratio = (value - min_val) / (max_val - min_val)
    
    b = int(color_low[0] + ratio * (color_high[0] - color_low[0]))
    g = int(color_low[1] + ratio * (color_high[1] - color_low[1]))
    r = int(color_low[2] + ratio * (color_high[2] - color_low[2]))
    
    return (b, g, r)


def clamp(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))


def moving_average(values, window=5):
    """
    Calculate moving average
    
    Args:
        values: list of values
        window: window size
    
    Returns:
        List of averaged values
    """
    if len(values) < window:
        return values
    
    averaged = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        averaged.append(np.mean(values[start:i+1]))
    
    return averaged


def get_file_exists(path):
    """Check if file exists and return path or None"""
    import os
    return path if os.path.exists(path) else None


def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers, returning default if denominator is zero"""
    if denominator == 0:
        return default
    return numerator / denominator