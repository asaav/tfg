import cv2
import numpy as np
import datetime


def scale_image(img, factor):
    height, width = img.shape[:2]
    scaledw = int(width * factor)
    scaledh = int(height * factor)
    return cv2.resize(img, (scaledw, scaledh), interpolation=cv2.INTER_CUBIC), scaledw, scaledh


def print_stats(img, w, h, start, time, total_time):
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - start)
    time = datetime.timedelta(seconds=round(time/1000))
    total_time = datetime.timedelta(seconds=total_time)
    stats = "FPS: {0}\nResolution: {1}x{2}\n{3} / {4}".format(fps, w, h, time, total_time)
    font = cv2.FONT_HERSHEY_PLAIN
    for i, line in enumerate(stats.split('\n')):
        cv2.putText(img, line, (10, 20+15*i), font, 0.8, (255, 255, 255))

    return img


# Malisiewicz et al.
def non_max_suppression(blobs, overlap_thresh):
    contours = np.array([blob[0] for blob in blobs])
    # if there are no boxes, return an empty list
    if len(contours) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if contours.dtype.kind == "i":
        contours = contours.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = contours[:, 0]
    y1 = contours[:, 1]
    x2 = contours[:, 2]
    y2 = contours[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    return [blobs[p] for p in pick]


def draw_contours(img, thresh):
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Delete contours with area less than 100
    contours = [c for c in contours if cv2.contourArea(c) > 100]

    # Delete contours with height/width more than 5
    aux = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if (h/w) < 3:
            aux.append(c)
    contours = aux

    # Create bounding boxes list with format (x1,y1,x2,y2)
    rects = np.array([cv2.boundingRect(c) for c in contours])
    rects[:, 2] += rects[:, 0]  # x2 = x1 + w
    rects[:, 3] += rects[:, 1]  # y2 = y1 + h

    # Non max suppression
    boundingBoxes = list(zip(rects, contours))
    picked = non_max_suppression(boundingBoxes, 0.3)
    contours = [p[1] for p in picked]  # get only contours

    # Calculate circularity using the formula C = (4*pi*area)/(perimeter^2)
    roundness = []
    for c in contours:
        roundness.append(4*np.pi*cv2.contourArea(c)/cv2.arcLength(c, True)**2)

    # Paint rectangles
    for index, c in enumerate(contours):
        if index == np.argmax(roundness) and np.max(roundness) > 0.45:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img
