import argparse
import os

import cv2
import numpy as np

import masks


def segment(image_path, mask, blur: bool = True, close: bool = True):
    """Read image, perform segmentation with given mask and return result."""
    image_bgr = cv2.imread(image_path)

    # Gaussian blur if high frequency noise reduction is needed
    if blur:
        image_bgr = cv2.GaussianBlur(image_bgr, (5, 5), cv2.BORDER_DEFAULT)

    # Convert to HSV
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Create mask
    masked = mask.mask(image_hsv)

    # Morphological closing for more noise reduction
    if close:
        masked = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # Apply mask
    segmented_image = cv2.bitwise_and(image_bgr, image_bgr, mask=masked)

    return segmented_image


if __name__ == "__main__":
    image_path = "test_img.png"

    image = segment(image_path, masks.Red)
    cv2.imshow("Preview", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image = segment(image_path, masks.Green)
    cv2.imshow("Preview", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image = segment(image_path, masks.Blue)
    cv2.imshow("Preview", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image = segment(image_path, masks.Magenta)
    cv2.imshow("Preview", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
