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
    elif method == "GMG":
        return GMGSubtractor()
    elif method == "CNT":
        return CNTSubtractor()
    elif method == 'GSOC':
        return GSOCSubtractor()
    else:
        return KNNSubtractor(500, 400, False)


class Subtractor (ABC):
    sub = None

    def getBackgroundImage(self):
        if self.sub is not None:
            return self.sub.getBackgroundImage()

    @abstractmethod
    def apply(self, image):
        pass


class GMGSubtractor (Subtractor):
    sub = None

    def __init__(self, initilizationFrames=24, decisionThreshold=0.8):
        self.sub = cv2.bgsegm.createBackgroundSubtractorGMG(initilizationFrames, decisionThreshold)

    def apply(self, image):
        if self.sub:
            fgmask = self.sub.apply(image)

            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(fgmask, cv2.MORPH_ERODE, kernel)
            return closed


class GSOCSubtractor (Subtractor):
    sub = None

    def __init__(self):
        self.sub = cv2.bgsegm.createBackgroundSubtractorGSOC()

    def apply(self, image):
        if self.sub:
            fgmask = self.sub.apply(image)

            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(fgmask, cv2.MORPH_ERODE, kernel)
            return closed


class CNTSubtractor(Subtractor):
    sub = None

    def __init__(self, minPixelStability=50, useHistory=True, maxPixelStability=900, isParallel=True):
        self.sub = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability, useHistory,
                                                            maxPixelStability, isParallel)

    def apply(self, image):
        if self.sub:
            fgmask = self.sub.apply(image)

            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            return closed


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

    def getBackgroundImage(self):
        return self.background

    def apply(self, image):
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        difference = cv2.absdiff(grey, self.background)
        thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((2, 2), np.uint8)
        erode = cv2.erode(thresh, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)

        return closed
