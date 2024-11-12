from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for reactive control
                            using the masked left lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    steer_matrix_left = np.random.rand(*shape)

    steer_matrix_left = np.ones(shape)
    steer_matrix_left[:,int(shape[1]/2):] = 0 * steer_matrix_left[:,int(shape[1]/2):]
    # ---
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for reactive control
                             using the masked right lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    print("$")
    steer_matrix_right = np.ones(shape)
    steer_matrix_right[:,:int(shape[1]/2)] = 0 * steer_matrix_right[:,:int(shape[1]/2)]
    # ---
    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    print("!")
    h, w, _ = image.shape
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply Gaussian blur to the grayscale image for noise reduction
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigma = 4
    img_gaussian_filter = cv2.GaussianBlur(img_gray,(0,0), sigma)
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    
    white_lower_hsv = np.array([0, 0, 140])         # CHANGE ME
    white_upper_hsv = np.array([250, 70, 255])   # CHANGE ME
    yellow_lower_hsv = np.array([15, 150, 110])        # CHANGE ME
    yellow_upper_hsv = np.array([35, 255, 255])  # CHANGE ME

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    STEER_LEFT_LM = get_steer_matrix_left_lane_markings((h,w))
    STEER_RIGHT_LM = get_steer_matrix_right_lane_markings((h,w))

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    threshold = 45 # CHANGE ME
    mask_mag = (Gmag > threshold)

    mask_left_edge = STEER_LEFT_LM * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = STEER_RIGHT_LM * mask_mag * mask_sobelx_pos * mask_sobely_neg

    # # # TODO: implement your own solution here
    # mask_left_edge = mask_yellow * STEER_LEFT_LM
    # mask_right_edge = mask_white * STEER_RIGHT_LM

    
    return mask_left_edge, mask_right_edge
