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
        # Use an image with 2x size of track window instead of full image
        im_height, im_width = image.shape[:2]
        (x, y, w, h) = self.track_window
        x = max(0, x-(w//2))
        y = max(0, y-(h//2))
        w = min(w*2, im_width)
        h = min(h*2, im_height)
        image = image[y:y+h, x:x+w]

        # Convert to RGB and calculate backprojection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dst = cv2.calcBackProject([image], [0, 1, 2], self.roi_hist, [0, 255, 0, 255, 0, 255], 1)

        # Get window relative to reduced image
        new_window = (self.track_window[0] - x, self.track_window[1] - y, self.track_window[2], self.track_window[3])

        # Apply meanshift
        ret, self.track_window = cv2.meanShift(dst, new_window, self.term_crit)

        # Coordinates are relative to reduced image, calculate them for full image
        x += self.track_window[0]
        y += self.track_window[1]
        w = self.track_window[2]
        h = self.track_window[3]
        self.track_window = (x, y, w, h)

        # Paint rectangle
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
