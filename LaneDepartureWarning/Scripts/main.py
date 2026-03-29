"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""

import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from IPython.display import HTML, Video
from moviepy import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *

class FindLaneLines:
    """ This class is for parameter tuning and lane detection.

    Attributes:
        calibration: Camera calibration object
        thresholding: Thresholding object
        transform: Perspective transformation object
        lanelines: Lane lines detection object
        left_indicator: State of left indicator (True = on)
        right_indicator: State of right indicator (True = on)
        show_overlay: Whether to show visual overlays on output
        img_size: Expected image size (width, height)
    """
    def __init__(self, camera_cal_path='../Images/camera_cal', show_overlay=False, img_size=(1280, 720)):
        """ Init Application
        
        Parameters:
            camera_cal_path (str): Path to camera calibration images
            show_overlay (bool): Whether to display overlays on output image
            img_size (tuple): Expected image size (width, height)
        """
        self.calibration = CameraCalibration(camera_cal_path, 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation(img_size=img_size)
        self.lanelines = LaneLines(img_size=img_size)
        self.left_indicator = False
        self.right_indicator = False
        self.show_overlay = show_overlay
        self.img_size = img_size
        
        # Store lane data without displaying
        self.lane_data = {
            'direction': None,
            'curvature': None,
            'position': None,
            'deviation_warning': False,
            'warning_message': None,
            'detection_failed': False
        }

    def set_indicator(self, left=False, right=False):
        """Set the state of turn indicators.
        
        Parameters:
            left (bool): Left indicator state
            right (bool): Right indicator state
        """
        self.left_indicator = left
        self.right_indicator = right

    def check_deviation_alert(self, position, direction):
        """Check if a lane departure alert should be issued.
        
        Parameters:
            position (float): Distance from center (negative = left, positive = right)
            direction (str): Current direction ('L', 'R', or 'F')
            
        Returns:
            tuple: (should_alert, warning_message)
        """
        # Threshold for deviation warning (meters)
        deviation_threshold = 0.5
        
        should_alert = False
        warning_message = None
        
        # Check if deviating left
        if position < -deviation_threshold:
            # Only alert if left indicator is NOT on
            if not self.left_indicator:
                should_alert = True
                warning_message = "WARNING: Lane Departure - Drifting Left"
        
        # Check if deviating right
        elif position > deviation_threshold:
            # Only alert if right indicator is NOT on
            if not self.right_indicator:
                should_alert = True
                warning_message = "WARNING: Lane Departure - Drifting Right"
        
        return should_alert, warning_message

    def forward(self, img):
        """Process a single frame through the lane detection pipeline.
        
        Parameters:
            img (np.array): Input image
            
        Returns:
            np.array: Processed output image
        """
        out_img = np.copy(img)
        
        # Resize if image size doesn't match expected
        if img.shape[1] != self.img_size[0] or img.shape[0] != self.img_size[1]:
            img = cv2.resize(img, self.img_size)
        
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        
        # Check if enough lane pixels were detected
        num_lane_pixels = np.count_nonzero(img)
        detection_failed = num_lane_pixels < 1000  # Threshold for minimum lane pixels
        
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        # Resize back to original size if needed
        if img.shape[1] != out_img.shape[1] or img.shape[0] != out_img.shape[0]:
            img = cv2.resize(img, (out_img.shape[1], out_img.shape[0]))

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        
        # Get lane data
        lR, rR, pos = self.lanelines.measure_curvature()
        direction = self.lanelines.get_direction()
        
        # Update stored data
        self.lane_data['direction'] = direction
        self.lane_data['curvature'] = min(lR, rR)
        self.lane_data['position'] = pos
        self.lane_data['detection_failed'] = detection_failed or not self.lanelines.has_valid_detection()
        
        # Only check for deviation if detection succeeded
        if not self.lane_data['detection_failed']:
            should_alert, warning_message = self.check_deviation_alert(pos, direction)
            self.lane_data['deviation_warning'] = should_alert
            self.lane_data['warning_message'] = warning_message
        else:
            self.lane_data['deviation_warning'] = False
            self.lane_data['warning_message'] = None
        
        # Only plot if overlay is enabled
        if self.show_overlay:
            out_img = self.lanelines.plot(out_img)
        
        return out_img

    def get_lane_data(self):
        """Get the current lane detection data.
        
        Returns:
            dict: Dictionary containing lane data and warning status
        """
        return self.lane_data.copy()

    def process_image(self, input_path, output_path):
        """Process a single image.
        
        Parameters:
            input_path (str): Path to input image
            output_path (str): Path to save output image
        """
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        """Process a video file.
        
        Parameters:
            input_path (str): Path to input video
            output_path (str): Path to save output video
        """
        clip = VideoFileClip(input_path)
        out_clip = clip.image_transform(self.forward)
        out_clip.write_videofile(output_path, audio=False)

    def process_frame(self, img):
        """Process a single frame (for external calling).
        
        Parameters:
            img (np.array): Input image frame
            
        Returns:
            tuple: (processed_image, lane_data_dict)
        """
        if img is None:
            raise ValueError("Input frame is None. Please provide a valid image.")
        
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Input frame must be a 3-channel color image. Got shape: {img.shape}")
        
        out_img = self.forward(img)
        return out_img, self.get_lane_data()


def main():
    args = docopt(__doc__)
    input_file = args['INPUT_PATH']
    output_file = args['OUTPUT_PATH']

    findLaneLines = FindLaneLines(show_overlay=True)
    
    if args['--video']:
        findLaneLines.process_video(input_file, output_file)
    else:
        findLaneLines.process_image(input_file, output_file)


if __name__ == "__main__":
    main()