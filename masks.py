from abc import ABC, abstractmethod

import numpy as np
import cv2 as cv

# All the HSV values were chosen experimentally


class Mask(ABC):
    @classmethod
    @abstractmethod
    def mask(cls, hsv_image):
        """Create a mask based on predefined hsv ranges."""


class Red(Mask):
    # Range for low hue
    low_1 = np.array([0, 125, 20])
    high_1 = np.array([10, 255, 255])
    # Range for high hue
    low_2 = np.array([170, 125, 20])
    high_2 = np.array([180, 255, 255])

    @classmethod
    def mask(cls, hsv_image):
        mask_1 = cv.inRange(hsv_image, cls.low_1, cls.high_1)
        mask_2 = cv.inRange(hsv_image, cls.low_2, cls.high_2)

        return mask_1 + mask_2


class Blue(Mask):
    low = np.array([94, 80, 2])
    high = np.array([126, 255, 255])

    @classmethod
    def mask(cls, hsv_image):
        return cv.inRange(hsv_image, cls.low, cls.high)


class Magenta(Mask):
    low = np.array([131, 95, 35])
    high = np.array([175, 255, 255])

    @classmethod
    def mask(cls, hsv_image):
        return cv.inRange(hsv_image, cls.low, cls.high)


class Green(Mask):
    low = np.array([17, 79, 19])
    high = np.array([76, 217, 153])

    @classmethod
    def mask(cls, hsv_image):
        return cv.inRange(hsv_image, cls.low, cls.high)
