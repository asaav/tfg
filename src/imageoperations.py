import cv2
import numpy as np
import datetime


def contour_distance(cont1, cont2):
    # Get both centers
    c1_m = cv2.moments(cont1)
    c1X = int(c1_m['m10'] / c1_m['m00'])
    c1Y = int(c1_m['m01'] / c1_m['m00'])

    c2_m = cv2.moments(cont2)
    c2X = int(c2_m['m10'] / c2_m['m00'])
    c2Y = int(c2_m['m01'] / c2_m['m00'])

    # calculate distance between centers and return it
    dx = c2X - c1X
    dy = c2Y - c1Y
    return np.sqrt(dx**2+dy**2)


max_id = 0


def match_contours(last_frame, current_frame, last_ids):
    global max_id
    # if there is no previous frame, return new ids
    if len(last_frame) == 0:
        return np.arange(len(current_frame))
    # find distance table
    distances = np.zeros(len(last_frame)*len(current_frame)).reshape(len(last_frame), len(current_frame))
    for i, contour1 in enumerate(last_frame):
        for j, contour2 in enumerate(current_frame):
            distances[i][j] = contour_distance(contour1, contour2)

    # find mins by col and row, if a row min is also its column min, they represent the same blob in different frames
    new_ids = [-1] * len(current_frame)
    for i in range(len(last_frame)):
        for j in range(len(current_frame)):
            col_min = np.min(distances[:, j])
            row_min = np.min(distances[i, :])
            if distances[i, j] == col_min and distances[i, j] == row_min:
                new_ids[j] = last_ids[i]
    for i, id in enumerate(new_ids):
        if id == -1:
            max_id += 1
            new_ids[i] = max_id
    return new_ids


def scale_image(img, factor):
    height, width = img.shape[:2]
    scaledw = int(width * factor)
    scaledh = int(height * factor)
    return cv2.resize(img, (scaledw, scaledh), interpolation=cv2.INTER_CUBIC), scaledw, scaledh


def print_stats(img, w, h, start, time, total_time):
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - start)
    time = datetime.timedelta(seconds=round(time/1000))
    total_time = datetime.timedelta(seconds=round(total_time))
    stats = "FPS: {0}\nResolution: {1}x{2}\n{3} / {4}\nPress H to get help".format(round(fps, 2), w, h, time, total_time)
    font = cv2.FONT_HERSHEY_PLAIN
    for i, line in enumerate(stats.split('\n')):
        cv2.putText(img, line, (10, 20+15*i), font, 1, (255, 255, 255))

    return img


# Malisiewicz et al.
def non_max_suppression(blobs, overlap_thresh):
    contours = np.array([blob[0] for blob in blobs])
    # if there are no boxes, return an empty list
    if len(contours) == 0:
        return []

    # if the bounding boxes integers, convert them to floats
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


def get_contours(subtractor_frame):
    im, contours, hierarchy = cv2.findContours(subtractor_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Delete contours with area less than 100
    contours = [c for c in contours if cv2.contourArea(c) > 100]

    # Delete contours with height/width more than 3 or width/height more than 2
    aux = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if (h / w) < 3:
            aux.append(c)
    contours = aux

    # Create bounding boxes list with format (x1,y1,x2,y2)
    rects = np.array([cv2.boundingRect(c) for c in contours])
    if len(rects) > 0:
        rects[:, 2] += rects[:, 0]  # x2 = x1 + w
        rects[:, 3] += rects[:, 1]  # y2 = y1 + h

        # Non max suppression
        boundingBoxes = list(zip(rects, contours))
        picked = non_max_suppression(boundingBoxes, 0.3)
        contours = [p[1] for p in picked]
    return contours


def draw_contours(img, ids, contours, ball_id=None):
    roundness = []
    if ball_id is None:
        # Calculate circularity using the formula C = (4*pi*area)/(perimeter^2)
        for c in contours:
            roundness.append(4 * np.pi * cv2.contourArea(c) / cv2.arcLength(c, True) ** 2)
        if roundness and np.max(roundness) > 0.70:
            ball_id = ids[np.argmax(roundness)]

    # Paint rectangles
    for index, c in enumerate(contours):
        if ids[index] == ball_id:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.putText(img, str(ids[index]), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            ball_id = ids[index]
        else:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.putText(img, str(ids[index]), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img, ball_id, roundness


def find_ball(ball_count, data):

    # Find minimum id for a contour marked as ball from those with more than 4 consecutive frames marked as ball
    print(len([value for value in data.values() if value[2]==754]))
    ball_ids = [int(key) for key, value in ball_count.items() if value >= 4 and key != 'None']
    ball_id = int(max(ball_count, key=ball_count.get))
    for d in data.values():
        if d[2] != ball_id:
            if d[2] in ball_ids:
                # Replace ball id with the min id
                d[0] = [ball_id if value == d[2] else value for value in d[0]]
                d[2] = ball_id
            elif d[2] not in ball_ids:
                # Check if one of the ids is already in our list
                contained = [value for value in ball_ids if value in d[0]]
                # if the id is not our min id, replace it
                if len(contained) == 1:
                    if contained[0] != ball_id:
                        d[0] = [ball_id if value == contained[0] else value for value in d[0]]
                    d[2] = ball_id

    locating_ball = False
    located_ball = None
    previous = None
    for k, d in data.items():
        # Get last frame tuple
        if str(int(k) - 1) in data:
            previous = data[str(int(k) - 1)]

        # If the ball doesn't appear anymore or the selected id appears, stop searching
        if located_ball is not None and (located_ball not in d[0] or ball_id in d[0]):
            located_ball = None
            locating_ball = False

        # If not currently searching and the selected ball id is not in current frame but it is in the last one,
        # start searching
        if previous is not None and ball_id not in d[0] and ball_id in previous[0] and not locating_ball:
            last_position = previous[1][previous[0].index(ball_id)]
            locating_ball = True
        if locating_ball and located_ball is None:
            # In subsequent frames, get every contour within 170 pixels and with more than 0.5 roundness
            # making sure they don't appear in the last frame
            near_ids = [possible_id for i, possible_id in enumerate(d[0])
                        if contour_distance(last_position, d[1][i]) < 170
                        and d[3][i] > 0.5
                        and possible_id not in previous[0]]
            # Once located, if we have more than one contour, get the one with more roundness (more likely to be ball)
            if len(near_ids) > 1:
                roundness = []
                for value in near_ids:
                    i = d[0].index(value)
                    roundness.append(d[3][i])
                located_ball = near_ids[np.argmax(roundness)]
            elif len(near_ids) == 1:
                located_ball = near_ids[0]
        # Replace id with the one selected by algorithm
        if located_ball is not None:
            d[0] = [ball_id if value == located_ball else value for value in d[0]]
            d[2] = ball_id
    print(len([value for value in data.values() if value[2] == 754]))
    print(len(data))
    return data
