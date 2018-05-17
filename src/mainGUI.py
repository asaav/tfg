import sys
import cv2
from PyQt5.QtGui import QImage, QPixmap
from qtpy import QtCore, QtGui

from comargs import gui_args
from imageoperations import scale_image, draw_contours, print_stats, match_contours, get_contours
from objectdetection import create_subtractor
from tracking import create_tracker

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QWidget, QLabel, QMainWindow, QAction, QFileDialog, QMessageBox, QPushButton, QFormLayout, \
    QApplication, QSlider


def init_trackers(rois, frame, method):
    trackers = []
    for roi in rois:
        tracker = create_tracker(method, roi, frame)
        trackers.append(tracker)

    return trackers


class VideoCapture(QWidget):
    def __init__(self, filename, subtraction, scale, tracker_method, parent):
        super(QWidget, self).__init__()
        self.videoFrame = QLabel()

        # Init timer
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.nextFrameSlot)

        # Opencv capture and stats
        self.cap = cv2.VideoCapture(str(filename[0]))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.length / self.cap.get(cv2.CAP_PROP_FPS)
        self.trackerMethod = tracker_method
        self.scale = scale

        # Program variables
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, self.duration)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.rawFrame = None
        self.trackers = []
        self.subtractor = create_subtractor(self.scale, subtraction)

        self.trackersB = QPushButton('Trackers', self)
        self.trackersB.setFixedWidth(50)
        self.trackersB.clicked.connect(self.init_trackers)

        self.last_ids = []
        self.last_contours = []

        parent.layout.addRow(self.positionSlider, self.trackersB)
        parent.layout.addRow(self.videoFrame)

    def setPosition(self, position):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, position * 1000)

    def init_trackers(self):
        self.timer.stop()
        winname = "Roi selection"
        rois = cv2.selectROIs(winname, img=self.rawFrame, fromCenter=False)
        cv2.destroyWindow(winname)
        for roi in rois:
            tracker = create_tracker(self.trackerMethod, roi, self.rawFrame)
            self.trackers.append(tracker)
        self.timer.start()

    def nextFrameSlot(self):
        start = cv2.getTickCount()

        # read frame
        ret, frame = self.cap.read()

        if ret:
            # scale image
            frame, width, height = scale_image(frame, args.scale)
            raw_frame = frame.copy()

            # operate with frame (tracking and subtraction)
            if len(self.trackers) > 0:

                for t in self.trackers:
                    t.update(raw_frame, frame)

            processed = self.subtractor.apply(raw_frame)

            contours = get_contours(processed)
            cont_ids = match_contours(self.last_contours, contours, self.last_ids)
            draw_contours(frame, cont_ids, contours)

            # add stats
            frame = print_stats(frame, width, height, start, self.cap.get(cv2.CAP_PROP_POS_MSEC), self.duration)
            self.last_ids = cont_ids
            self.last_contours = contours

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            self.positionSlider.setValue(int(round(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)))
            self.videoFrame.setPixmap(pix)

    def start(self):
        # Start timer with timeout of 1000/fps ms
        self.timer.start(1000.0/self.frame_rate)

    def pause(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QWidget, self).deleteLater()


class ControlWindow(QMainWindow):
    def __init__(self, args):
        super(ControlWindow, self).__init__()
        self.setGeometry(50, 50, 800, 600)
        self.setWindowTitle("PyTrack")

        self.capture = None

        self.isVideoFileLoaded = False
        self.videoFileName = None

        self.quitAction = QAction("Salir", self)
        self.quitAction.setShortcut("Ctrl+Q")
        self.quitAction.triggered.connect(self.closeApplication)

        self.openVideoFile = QAction("Abrir vídeo", self)
        self.openVideoFile.setShortcut("Ctrl+Shift+V")
        self.openVideoFile.triggered.connect(self.loadVideoFile)

        self.mainMenu = self.menuBar()
        self.fileMenu = self.mainMenu.addMenu('Archivo')
        self.fileMenu.addAction(self.openVideoFile)
        self.fileMenu.addAction(self.quitAction)

        self.wid = QWidget(self)
        self.setCentralWidget(self.wid)
        self.layout = QFormLayout(self)
        self.startButton = QPushButton('Play', self)
        self.startButton.clicked.connect(self.startCapture)
        self.startButton.setFixedWidth(50)
        self.pauseButton = QPushButton('Pause', self)
        self.pauseButton.setFixedWidth(50)

        self.layout.addRow(self.startButton, self.pauseButton)

        self.wid.setLayout(self.layout)

    def keyPressEvent(self, a0: QtGui.QKeyEvent):
        if a0.key() == QtCore.Qt.Key_T:
            if self.capture is not None and self.isVideoFileLoaded:
                self.capture.init_trackers()

    def setPosition(self, position):
        self.capture.setPosition(position)

    def startCapture(self):
        if self.videoFileName is not None:
            if not self.capture and self.isVideoFileLoaded:
                self.capture = VideoCapture(self.videoFileName, args.backsub, args.scale, args.tracker, self)
                self.pauseButton.clicked.connect(self.capture.pause)
            self.capture.start()

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None

    def loadVideoFile(self):
        self.videoFileName = QFileDialog.getOpenFileName(self, 'Select a Video File', filter="Video (*.avi *.mp4)")
        self.isVideoFileLoaded = True

    def closeApplication(self):
        choice = QMessageBox.question(self, 'Message', 'Do you really want to exit?', QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            print("Closing....")
            sys.exit()
        else:
            pass


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    args = gui_args()
    sys.excepthook = except_hook
    app = QApplication(sys.argv)
    window = ControlWindow(args)
    window.show()
    sys.exit(app.exec_())
