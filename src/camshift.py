import cv2
import numpy as np


def scale_image(img, factor):
    height, width = img.shape[:2]
    scaledw = int(width * factor)
    scaledh = int(height * factor)
    return cv2.resize(img, (scaledw, scaledh), interpolation=cv2.INTER_CUBIC)


cap = cv2.VideoCapture("C:/Users/anton/Downloads/VideosVolley/segunda-tanda/prueba11_1920.avi")

# take first frame of the video
ret, frame = cap.read()
if ret:
    frame = scale_image(frame, 0.75)

# setup initial location of window
roi = cv2.selectROI("ROI", frame, False)
cv2.destroyWindow("ROI")

track_window = (roi[0], roi[1], roi[2], roi[3])
# set up the ROI for tracking
roi = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
cv2.imshow("roi", roi)
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
# mask = cv2.inRange(rgb_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 10)
while True:
    ret, frame = cap.read()
    if ret:
        frame = scale_image(frame, 0.75)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dst = cv2.calcBackProject([hsv], [0, 1, 2], roi_hist, [0, 255, 0, 255, 0, 255], 1)

        # apply Camshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)

        # # apply meanshift to get the new location
        # ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # # Draw it on image
        # x, y, w, h = track_window
        # img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow('backprojection', dst)
        cv2.imshow('img2', img2)

        k = cv2.waitKey(25) & 0xff
        if k == 27:
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()
