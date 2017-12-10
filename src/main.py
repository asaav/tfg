import numpy as np
import cv2
import sys
import objectdetection

substractor = objectdetection.MOGSubtractor(200, 5, 0.7, 0)


def scale_image(img, factor):
    height, width = img.shape[:2]
    scaledw = int(width * factor)
    scaledh = int(height * factor)
    return cv2.resize(img, (scaledw, scaledh), interpolation=cv2.INTER_CUBIC), scaledw, scaledh


def print_stats(img, w, h, start):
    fps = cv2.getTickFrequency()/(cv2.getTickCount() - start)
    stats = "FPS: {0}\nResolution: {1}x{2}".format(fps, w, h)
    font = cv2.FONT_HERSHEY_PLAIN
    for i, line in enumerate(stats.split('\n')):
        cv2.putText(img, line, (10, 20+15*i), font, 0.8, (255, 255, 255))

    return img


def background_difference(foreground, scale):
    background = cv2.imread('background.jpg', 0)
    background, w, h = scale_image(background, scale)

    grey = cv2.cvtColor(foreground, cv2.COLOR_RGB2GRAY)

    difference = cv2.absdiff(grey, background)
    thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((2, 2), np.uint8)
    erode = cv2.erode(thresh, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)

    return closed


def background_MOG(foreground):
    global substractor
    if not substractor:
        substractor = cv2.bgsegm.createBackgroundSubtractorMOG(200, 5, 0.7, 0)

    fgmask = substractor.apply(foreground)

    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    return closed


def background_MOG2(foreground):
    global substractor
    if not substractor:
        substractor = cv2.createBackgroundSubtractorMOG2(500, 50, False)

    fgmask = substractor.apply(foreground)

    kernel = np.ones((4, 4), np.uint8)
    closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    return closed


def background_KNN(foreground):
    global substractor
    if not substractor:
        substractor = cv2.createBackgroundSubtractorKNN(500, 400, False)

    fgmask = substractor.apply(foreground)

    kernel = np.ones((4, 4), np.uint8)
    closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    return closed


def sub_background(foreground, scale, method):
    if method == "resta":
        return background_difference(foreground, scale)
    elif method == "MOG":
        return substractor.apply(foreground)
    elif method == "MOG2":
        return background_MOG2(foreground)
    else:
        return background_KNN(foreground)


def draw_contours(img, thresh):
    contoursim = thresh.copy()
    im, contours, hierarchy = cv2.findContours(contoursim, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def main():
    video = sys.argv[1]
    scale = float(sys.argv[2])
    if len(sys.argv) > 3:
        method = sys.argv[3]
    else:
        method = "resta"
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print sys.argv[1] + " couldn't be opened."

    while cap.isOpened():
        start = cv2.getTickCount()

        # read frame
        ret, frame = cap.read()

        if ret:
            # scale image
            frame, width, height = scale_image(frame, scale)

            # operate with frame
            processed = sub_background(frame, scale, method)

            draw_contours(frame, processed)
            # add stats
            processed = print_stats(processed, width, height, start)

            cv2.imshow('processed', processed)
            cv2.imshow('original', frame)

        # end video if q is pressed or no frame was read
        if (cv2.waitKey(30) & 0xFF == ord('q')) or (not ret):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
