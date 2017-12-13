import numpy as np
import cv2
import sys
from objectdetection import MOG2Subtractor, MOGSubtractor, KNNSubtractor, DifferenceSubtractor


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


def create_subtractor(scale, method):
    if method == "resta":
        background = cv2.imread('background.jpg', 0)
        background, w, h = scale_image(background, scale)
        return DifferenceSubtractor(background)
    elif method == "MOG":
        return MOGSubtractor(200, 5, 0.7, 0)
    elif method == "MOG2":
        return MOG2Subtractor(500, 50, True)
    else:
        return KNNSubtractor(500, 400, False)


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
        method = "MOG2"
    cap = cv2.VideoCapture(video)

    subtractor = create_subtractor(scale, method)

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
            processed = subtractor.apply(frame)

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
