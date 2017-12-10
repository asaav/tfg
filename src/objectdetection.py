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
