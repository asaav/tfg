import os
import pickle
import sys
import cv2

from dialogue import HelpDialogue
from imageoperations import scale_image, draw_contours, print_stats, match_contours, get_contours
from objectdetection import create_subtractor
from tracking import create_tracker
from comargs import process_args
from PyQt5 import QtWidgets


def init_trackers(rois, frame, method):
    trackers = []
    for roi in rois:
        tracker = create_tracker(method, roi, frame)
        trackers.append(tracker)

    return trackers


def main():
    args = process_args()

    cap = cv2.VideoCapture(args.video)
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    trackers = []
    subtractor = create_subtractor(args.backsub)
    log = open('{0}.log'.format(video_name), 'w')
    log.write("FRAME NUMBER;POSITION;ID;BALL\n")
    if not cap.isOpened():
        print(sys.argv[1] + " couldn't be opened.",
              file=sys.stderr)
        exit(1)

    action = False
    play_video = True
    ret = None
    raw_frame = None
    last_contours = []
    last_ids = []
    all_contours = []

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
                        t.update(raw_frame, frame)

                processed = subtractor.apply(raw_frame)

                contours = get_contours(processed)
                cont_ids = match_contours(last_contours, contours, last_ids)
                frame, ball_id = draw_contours(frame, cont_ids, contours)

                if action:
                    nframe = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    for i, c in enumerate(contours):
                        (x, y, w, h) = cv2.boundingRect(c)
                        cont_id = cont_ids[i]
                        log.write('{0};{1};{2};{3}\n'.format(nframe, (x, y), cont_id, cont_id == ball_id))
                        all_contours.append((nframe, contours, contours, ball_id))

                # add stats
                frame = print_stats(frame, width, height, start, cap.get(cv2.CAP_PROP_POS_MSEC), video_length)
                last_ids = cont_ids
                last_contours = contours
                cv2.imshow('processed', processed)
                cv2.imshow('original', frame)

        # end video if q is pressed or no frame was read
        key = cv2.waitKey(20)
        if (key == ord('q')) or (not ret):
            break
        # if t is pressed, open window to select roi
        elif key == ord('t'):
            winname = "Roi selection"
            rois = cv2.selectROIs(winname, img=raw_frame, fromCenter=False)
            trackers = init_trackers(rois, raw_frame, args.tracker)
            cv2.destroyWindow(winname)
        # space to pause
        elif key == ord(' '):
            play_video = not play_video
        # j to start/stop an action
        elif key == ord('j'):
            action = not action
        # k to rewind 5 seconds
        elif key == ord('k'):
            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_MSEC, time - 5000)
        # l to forward 5 seconds
        elif key == ord('l'):
            time = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_MSEC, time + 5000)
        # h to get help window
        elif key == ord('h'):
            window = HelpDialogue(False)
            window.show()

    cap.release()
    dmp = open('{0}.dmp'.format(video_name), 'wb')
    pickle.dump(all_contours, dmp)
    log.close()
    dmp.close()
    cv2.imwrite('backgroundModel.jpg', subtractor.getBackgroundImage())
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main()
    app.exit()
