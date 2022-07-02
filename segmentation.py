import argparse
import os

import cv2 as cv
import numpy as np

import masks


def segment(image_path: str, mask: masks.Mask, blur: bool = True, close: bool = True):
    """Read RGB image, perform segmentation with given mask and return result."""

    image_rgb = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)

    # Gaussian blur if high frequency noise reduction is needed
    if blur:
        image_rgb = cv.GaussianBlur(image_rgb, (5, 5), cv.BORDER_DEFAULT)

    # Convert to HSV
    image_hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)

    # Create mask
    masked = mask.mask(image_hsv)

    # Morphological closing for more noise reduction
    if close:
        masked = cv.morphologyEx(masked, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # Apply mask
    segmented_image = cv.bitwise_and(image_rgb, image_rgb, mask=masked)

    return segmented_image
