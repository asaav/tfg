import numpy as np
import cv2


def scale_image(img, factor):
    height, width = img.shape[:2]
    scaledw = int(width * factor)
    scaledh = int(height * factor)
    return cv2.resize(img, (scaledw, scaledh), interpolation=cv2.INTER_CUBIC)


cap = cv2.VideoCapture('../videos/rodando.avi')

# take first frame of the video
ret, frame = cap.read()
if ret:
    frame = scale_image(frame, 0.5)

# setup initial location of window
roi = cv2.selectROI("ROI", frame, False)

track_window = (roi[0], roi[1], roi[2], roi[3])
# set up the ROI for tracking
roi = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
cv2.imshow("roi", roi)
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 10)
while True:
    ret, frame = cap.read()
    frame = scale_image(frame, 0.5)
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # # apply meanshift to get the new location
        # ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # # Draw it on image
        # pts = cv2.boxPoints(ret)
        # pts = np.int0(pts)
        # img2 = cv2.polylines(frame,[pts],True, 255,2)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow('img2', img2)

        cv2.imshow('img2',img2)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()
