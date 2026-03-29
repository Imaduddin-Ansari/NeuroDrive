import cv2
import numpy as np

class PerspectiveTransformation:
    """ This a class for transforming image between front view and top view

    Attributes:
        src (np.array): Coordinates of 4 source points
        dst (np.array): Coordinates of 4 destination points
        M (np.array): Matrix to transform image from front view to top view
        M_inv (np.array): Matrix to transform image from top view to front view
    """
    def __init__(self, img_size=(1280, 720)):
        """Init PerspectiveTransformation.
        
        Parameters:
            img_size (tuple): Image size (width, height)
        """
        self.img_size = img_size
        width, height = img_size
        
        # Scale transformation points based on image size
        # Original points were for 1280x720
        scale_x = width / 1280.0
        scale_y = height / 720.0
        
        # Original source points (for 1280x720)
        orig_src = np.float32([
            (550, 460),     # top-left
            (150, 720),     # bottom-left
            (1200, 720),    # bottom-right
            (770, 460)      # top-right
        ])
        
        # Scale source points
        self.src = np.float32([
            (550 * scale_x, 460 * scale_y),     # top-left
            (150 * scale_x, height),            # bottom-left
            (1200 * scale_x, height),           # bottom-right
            (770 * scale_x, 460 * scale_y)      # top-right
        ])
        
        # Scale destination points
        self.dst = np.float32([
            (100 * scale_x, 0),
            (100 * scale_x, height),
            (1100 * scale_x, height),
            (1100 * scale_x, 0)
        ])
        
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward(self, img, flags=cv2.INTER_LINEAR):
        """ Take a front view image and transform to top view

        Parameters:
            img (np.array): A front view image
            flags : flag to use in cv2.warpPerspective()

        Returns:
            Image (np.array): Top view image
        """
        return cv2.warpPerspective(img, self.M, self.img_size, flags=flags)

    def backward(self, img, flags=cv2.INTER_LINEAR):
        """ Take a top view image and transform it to front view

        Parameters:
            img (np.array): A top view image
            flags (int): flag to use in cv2.warpPerspective()

        Returns:
            Image (np.array): Front view image
        """
        return cv2.warpPerspective(img, self.M_inv, self.img_size, flags=flags)