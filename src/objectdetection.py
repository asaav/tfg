import cv2
import numpy as np
from abc import ABC, abstractmethod


def create_subtractor(method):
    if method == "resta":
        background = cv2.imread('backgroundModel1.jpg')
        background = cv2.GaussianBlur(background, (5, 5), 0)
        return DifferenceSubtractor(background)
    elif method == "MOG":
        return MOGSubtractor(200, 5, 0.55, 8)
    elif method == "MOG2":
        return MOG2Subtractor(250, 40, True)
    elif method == "GMG":
        return GMGSubtractor(120, 0.90)
    elif method == "CNT":
        return CNTSubtractor(minPixelStability=30, maxPixelStability=180, isParallel=True)
    elif method == 'GSOC':
        return GSOCSubtractor()
    else:
        return KNNSubtractor(350, 400, True)


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

    def __init__(self, initilizationFrames=120, decisionThreshold=0.8):
        self.sub = cv2.bgsegm.createBackgroundSubtractorGMG(initilizationFrames, decisionThreshold)

    def apply(self, image):
        if self.sub:
            fgmask = self.sub.apply(image)

            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            return fgmask


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

    def __init__(self, minPixelStability=15, useHistory=True, maxPixelStability=900, isParallel=True):
        self.sub = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability, useHistory,
                                                            maxPixelStability, isParallel)

    def apply(self, image):
        if self.sub:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            fgmask = self.sub.apply(image)

            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            return fgmask


class MOGSubtractor (Subtractor):
    sub = None

    def __init__(self, history=200, nmixtures=5, backgroundratio=0.7, noisesigma=0.0):
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
            fgmask = self.sub.apply(image, learningRate=0.0006)

            # remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

            # remove shadows
            thresh = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)[1]

            return thresh


class KNNSubtractor (Subtractor):
    sub = None

    def __init__(self, history=500, dist2threshold=400, detectshadows=True):
        self.sub = cv2.createBackgroundSubtractorKNN(history, dist2threshold, detectshadows)

    def apply(self, image):
        if self.sub:
            fgmask = self.sub.apply(image)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            # kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

            # remove shadows
            thresh = cv2.threshold(fgmask, 170, 255, cv2.THRESH_BINARY)[1]

            return thresh


class DifferenceSubtractor (Subtractor):
    __background = None

    def __init__(self, background):
        self.__background = background

    def getBackgroundImage(self):
        return self.__background

    def apply(self, image):
        # blur image to delete noise
        image = cv2.GaussianBlur(image, (5, 5), 0)
        difference = cv2.absdiff(self.__background, image)

        lower = (35, 35, 35)
        upper = (255, 255, 255)
        thresh = cv2.inRange(difference, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # kernel = np.ones((3, 3), np.uint8)
        # thresh = cv2.erode(thresh, kernel, iterations=1)
        #
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return thresh
