import cv2
import numpy as np
from abc import ABC, abstractmethod


def create_tracker(method, roi, frame):
    if method == "meanshift":
        return MeanShiftTracker(roi, frame)
    elif method == "camshift":
        return CamShiftTracker(roi, frame)


class Tracker (ABC):

    track_window = None
    roi_hist = None
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 10)

    @abstractmethod
    def __init__(self, roi, frame):
        self.track_window = (roi[0], roi[1], roi[2], roi[3])
        region = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        rgb_roi = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        self.roi_hist = cv2.calcHist([rgb_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

    @abstractmethod
    def update(self, image, drawto):
        pass


class MeanShiftTracker (Tracker):

    def __init__(self, roi, frame):
        super().__init__(roi, frame)

    def update(self, image, drawto):
        # Convert to RGB and calculate backprojection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dst = cv2.calcBackProject([image], [0, 1, 2], self.roi_hist, [0, 255, 0, 255, 0, 255], 1)

        # Apply meanshift
        ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)

        # Paint rectangle
        x, y, w, h = self.track_window

        drawto = cv2.rectangle(drawto, (x, y), (x + w, y + h), 255, 2)
        return drawto


class CamShiftTracker (Tracker):
    def __init__(self, roi, frame):
        super().__init__(roi, frame)

    def update(self, image, drawto):
        # Convert to RGB and calculate backprojection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dst = cv2.calcBackProject([image], [0, 1, 2], self.roi_hist, [0, 255, 0, 255, 0, 255], 1)

        # Apply meanshift
        ret, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)

        # Paint rectangle
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        drawto = cv2.polylines(drawto, [pts], True, 255, 2)
        return drawto
