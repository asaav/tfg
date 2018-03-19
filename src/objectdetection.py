import cv2
import numpy as np
from imageoperations import scale_image
from abc import ABC, abstractmethod


def create_subtractor(scale, method):
    if method == "resta":
        background = cv2.imread('background.jpg', 0)
        background, w, h = scale_image(background, scale)
        return DifferenceSubtractor(background)
    elif method == "MOG":
        return MOGSubtractor(200, 5, 0.7, 0)
    elif method == "MOG2":
        return MOG2Subtractor(500, 50, True)
    else:
        return KNNSubtractor(500, 400, False)


class Subtractor (ABC):

    @abstractmethod
    def apply(self, image):
        pass


class MOGSubtractor (Subtractor):
    sub = None

    def __init__(self, history=None, nmixtures=None, backgroundratio=None, noisesigma=None):
        self.sub = cv2.bgsegm.createBackgroundSubtractorMOG(history, nmixtures, backgroundratio, noisesigma)

    def apply(self, image):
        if self.sub:
            fgmask = self.sub.apply(image)

            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            return closed


class MOG2Subtractor (Subtractor):
    sub = None

    def __init__(self, history=None, varthreshold=None, detectshadows=None):
        self.sub = cv2.createBackgroundSubtractorMOG2(history, varthreshold, detectshadows)

    def apply(self, image):
        if self.sub:
            fgmask = self.sub.apply(image)

            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            return closed


class KNNSubtractor (Subtractor):
    sub = None

    def __init__(self, history=None, dist2threshold=None, detectshadows=None):
        self.sub = cv2.createBackgroundSubtractorKNN(history, dist2threshold, detectshadows)

    def apply(self, image):
        if self.sub:
            fgmask = self.sub.apply(image)

            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            return closed


class DifferenceSubtractor (Subtractor):

    background = None

    def __init__(self, background):
        self.background = background

    def apply(self, image):
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        difference = cv2.absdiff(grey, self.background)
        thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((2, 2), np.uint8)
        erode = cv2.erode(thresh, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)

        return closed
