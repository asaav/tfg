import numpy as np
import cv2
import sys
import time


def scale_frame(img, factor):
    height, width = img.shape[:2]
    scaledw = int(width * factor)
    scaledh = int(height * factor)
    return cv2.resize(img, (scaledw, scaledh), interpolation=cv2.INTER_CUBIC), scaledw, scaledh


def print_stats(img, w, h, start, count):
    fps = count/(time.time() - start)
    stats = "FPS: {0}\nResolution: {1}x{2}".format(fps, w, h)
    font = cv2.FONT_HERSHEY_PLAIN
    for i, line in enumerate(stats.split('\n')):
        cv2.putText(img, line, (10, 20+15*i), font, 0.8, (255, 255, 255))

    return img


def main():
    video = sys.argv[1]
    scale = float(sys.argv[2])
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print sys.argv[1] + " couldn't be opened."

    start = time.time()
    fpscount = 0

    while cap.isOpened():
        # rqead frame
        ret, frame = cap.read()

        if ret:
            # scale image
            frame, width, height = scale_frame(frame, scale)

            # operate with frame

            # add stats
            fpscount += 1
            frame = print_stats(frame, width, height, start, fpscount)

            cv2.imshow('frame', frame)

        # end video if q is pressed or no frame was read
        if (cv2.waitKey(30) & 0xFF == ord('q')) or (not ret):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
