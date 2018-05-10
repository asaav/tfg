import sys
import cv2
from imageoperations import scale_image, draw_contours, print_stats
from objectdetection import create_subtractor
from tracking import create_tracker
from comargs import process_args


def init_trackers(rois, frame, method):
    trackers = []
    for roi in rois:
        tracker = create_tracker(method, roi, frame)
        trackers.append(tracker)

    return trackers


def main():

    args = process_args()

    cap = cv2.VideoCapture(args.video)
    video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)
    trackers = []
    subtractor = create_subtractor(args.scale, args.backsub)

    if not cap.isOpened():
        print(sys.argv[1] + " couldn't be opened.",
              file=sys.stderr)
        exit(1)

    play_video = True
    ret = None
    raw_frame = None

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

                draw_contours(frame, processed)

                # add stats
                frame = print_stats(frame, width, height, start, cap.get(cv2.CAP_PROP_POS_MSEC), video_length)

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
            trackers = init_trackers(rois, raw_frame, args.tracker)
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
