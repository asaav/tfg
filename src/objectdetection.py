import cv2
import numpy as np
from abc import ABC, abstractmethod


def create_subtractor(method):
    if method == "resta":
        background = cv2.imread('backgroundModel1.jpg')
        background = cv2.GaussianBlur(background, (5, 5), 0)
        return DifferenceSubtractor(background)
    elif method == "MOG":
        return MOGSubtractor(200, 5, 0.7, 0)
    elif method == "MOG2":
        return MOG2Subtractor(500, 60, True)
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

    def __init__(self, history=500, varthreshold=50, detectshadows=True):
        self.sub = cv2.createBackgroundSubtractorMOG2(history, varthreshold, detectshadows)

    def apply(self, image):
        if self.sub:
            fgmask = self.sub.apply(image, learningRate=0.0005)

            # remove noise
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

            # remove shadows
            thresh = cv2.threshold(closed, 128, 255, cv2.THRESH_BINARY)

            return thresh[1]


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
    __background = None

    def __init__(self, background):
        self.__background = background

    def getBackgroundImage(self):
        return self.__background

    def apply(self, image):
        # blur image to delete noise
        image = cv2.GaussianBlur(image, (5, 5), 0)
        difference = cv2.absdiff(image, self.__background)

        lower = (25, 25, 25)
        upper = (255, 255, 255)
        thresh = cv2.inRange(difference, lower, upper)

        # kernel = np.ones((3, 3), np.uint8)
        # thresh = cv2.erode(thresh, kernel, iterations=1)
        #
        # kernel = np.ones((3, 3), np.uint8)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return thresh
