import cv2
import numpy as np
from abc import ABCMeta, abstractmethod


class Subtractor (object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, image):
        pass


class MOGSubtractor (Subtractor):
    sub = None

    def __init__(self, history=None, nmixtures=None, backgroundRatio=None, noiseSigma=None):
        self.sub = cv2.bgsegm.createBackgroundSubtractorMOG(history, nmixtures, backgroundRatio, noiseSigma)

    def apply(self, image):
        if self.sub:
            fgmask = self.sub.apply(image)

            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            return closed


class MOG2Subtractor (Subtractor):
    sub = None

    def __init__(self, history=None, varThreshold=None, detectShadows=None):
        self.sub = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)

    def apply(self, image):
        if self.sub:
            fgmask = self.sub.apply(image)

            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            return closed


class KNNSubtractor (Subtractor):
    sub = None

    def __init__(self, history=None, dist2Threshold=None, detectShadows=None):
        self.sub = cv2.createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows)

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


class SubtractorFactory:
    def get_person(self, kind, arg1=None, arg2=None, arg3=None, arg4=None):
        if kind == "resta":
            return DifferenceSubtractor(arg1)
        elif kind == "MOG":
            return MOGSubtractor(arg1, arg2, arg3, arg4)
        elif kind == "MOG2":
            return MOG2Subtractor(arg1, arg2, arg3)
        elif kind == "KNN":
            return KNNSubtractor(arg1, arg2, arg3)
        else:
            raise NotImplementedError("Unknown person type.")
