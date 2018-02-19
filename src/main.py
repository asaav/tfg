import argparse
import sys
import cv2
import numpy as np
from objectdetection import create_subtractor


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def scale_image(img, factor):
    height, width = img.shape[:2]
    scaledw = int(width * factor)
    scaledh = int(height * factor)
    return cv2.resize(img, (scaledw, scaledh), interpolation=cv2.INTER_CUBIC), scaledw, scaledh


def print_stats(img, w, h, start):
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - start)
    stats = "FPS: {0}\nResolution: {1}x{2}".format(fps, w, h)
    font = cv2.FONT_HERSHEY_PLAIN
    for i, line in enumerate(stats.split('\n')):
        cv2.putText(img, line, (10, 20+15*i), font, 0.8, (255, 255, 255))

    return img


def draw_contours(img, thresh):
    contoursim = thresh.copy()
    im, contours, hierarchy = cv2.findContours(contoursim, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    roundness = []
    for c in contours:
        if cv2.contourArea(c) < 100:
            roundness.append(0)
        else:
            roundness.append(4*np.pi*cv2.contourArea(c)/cv2.arcLength(c, 1)**2)

    for index, c in enumerate(contours):
        if cv2.contourArea(c) < 100:
            continue
        if index == np.argmax(roundness) and np.max(roundness) > 0.45:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def init_trackers(rois, frame):
    trackers = []
    for roi in rois:
        tracker = cv2.TrackerTLD_create()
        tracker.init(frame, (roi[0], roi[1], roi[2], roi[3]))
        trackers.append(tracker)

    return trackers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="video to be processed")
    parser.add_argument("-s", "--scale", type=restricted_float, help="reduction factor")
    parser.add_argument("-m", "--method", choices=['resta', 'MOG2', 'MOG', 'KNN'], default='MOG2')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    trackers = []
    subtractor = create_subtractor(args.scale, args.method)

    if not cap.isOpened():
        print(sys.argv[1] + " couldn't be opened.",
              file=sys.stderr)
        exit(1)

    play_video = True

    while cap.isOpened():
        if play_video:
            start = cv2.getTickCount()

            # read frame
            ret, frame = cap.read()

            if ret:
                # scale image
                frame, width, height = scale_image(frame, args.scale)
                raw_frame = frame.copy()

                # operate with frame (tracking and subtraction)
                if len(trackers) > 0:
                    for t in trackers:
                        ok, bbox = t.update(raw_frame)
                        if ok:
                            p1 = (int(bbox[0]), int(bbox[1]))
                            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                            cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
                        else:
                            # Tracking failure
                            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        (0, 0, 255), 2)

                processed = subtractor.apply(raw_frame)

                draw_contours(frame, processed)

                # add stats
                processed = print_stats(processed, width, height, start)

                cv2.imshow('processed', processed)
                cv2.imshow('original', frame)

        # end video if q is pressed or no frame was read
        key = cv2.waitKey(5)
        if (key == ord('q')) or (not ret):
            break
        # if t is pressed, open window to select roi
        elif key == ord('t'):
            winname = "Roi selection"
            rois = cv2.selectROIs(winname, img=raw_frame, fromCenter=False)
            trackers = init_trackers(rois, raw_frame)
            cv2.destroyWindow(winname)
        # space to pause
        elif key == ord(' '):
            play_video = not play_video
        # j to rewind 5 seconds
        elif key == ord('j'):
            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_MSEC, time-5000)
        elif key == ord('k'):
            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_MSEC, time+5000)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
