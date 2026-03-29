import cv2
import numpy as np

def threshold_rel(img, lo, hi):
    vmin = np.min(img)
    vmax = np.max(img)
    
    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255

class Thresholding:
    """ This class is for extracting relevant pixels in an image.
    """
    def __init__(self):
        """ Init Thresholding."""
        pass

    def forward(self, img):
        """ Take an image and extract all relevant pixels.

        Parameters:
            img (np.array): Input image

        Returns:
            binary (np.array): A binary image represent all positions of relevant pixels.
        """
        height, width = img.shape[:2]
        
        # Convert to different color spaces
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        v_channel = hsv[:,:,2]
        b_channel = lab[:,:,2]
        
        # White lane detection (works for right lanes, usually white)
        # Use L channel for brightness
        white_lane = threshold_rel(l_channel, 0.8, 1.0)
        # Also use V channel
        white_lane2 = threshold_abs(v_channel, 200, 255)
        white_combined = cv2.bitwise_or(white_lane, white_lane2)
        
        # Mask to right side (adaptive to resolution)
        white_combined[:, :int(width*0.4)] = 0
        
        # Yellow lane detection (works for left lanes, usually yellow)
        # Method 1: HSV for yellow
        yellow_lane = threshold_abs(h_channel, 15, 35)
        yellow_lane &= threshold_abs(s_channel, 40, 255)
        yellow_lane &= threshold_abs(v_channel, 50, 255)
        
        # Method 2: LAB color space (B channel good for yellow)
        yellow_lab = threshold_abs(b_channel, 145, 200)
        
        # Combine yellow detection methods
        yellow_combined = cv2.bitwise_or(yellow_lane, yellow_lab)
        
        # Mask to left side (adaptive to resolution)
        yellow_combined[:, int(width*0.55):] = 0
        
        # Sobel edge detection for additional robustness
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sobel_binary = threshold_abs(scaled_sobel, 20, 100)
        
        # S channel thresholding (good for both yellow and white)
        s_binary = threshold_abs(s_channel, 170, 255)
        
        # Combine all methods
        # Priority: Color-specific detection, then S channel, then edges
        img_combined = cv2.bitwise_or(yellow_combined, white_combined)
        img_combined = cv2.bitwise_or(img_combined, s_binary)
        
        # Add edge detection but with less weight (only where we don't have color detection)
        edge_mask = cv2.bitwise_and(sobel_binary, cv2.bitwise_not(img_combined))
        img_combined = cv2.bitwise_or(img_combined, edge_mask)
        
        # Clean up noise with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        img_combined = cv2.morphologyEx(img_combined, cv2.MORPH_OPEN, kernel)
        img_combined = cv2.morphologyEx(img_combined, cv2.MORPH_CLOSE, kernel)
        
        return img_combined