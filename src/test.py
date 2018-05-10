import sys
import cv2
from PyQt5.QtGui import QImage, QPixmap
from qtpy import QtCore, QtGui

from imageoperations import scale_image, draw_contours, print_stats
from objectdetection import create_subtractor
from tracking import create_tracker

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QWidget, QLabel, QMainWindow, QAction, QFileDialog, QMessageBox, QPushButton, QFormLayout, \
    QApplication


def init_trackers(rois, frame, method):
    trackers = []
    for roi in rois:
        tracker = create_tracker(method, roi, frame)
        trackers.append(tracker)

    return trackers


class VideoCapture(QWidget):
    def __init__(self, filename, parent):
        super(QWidget, self).__init__()
        self.cap = cv2.VideoCapture(str(filename[0]))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_frame = QLabel()
        self.raw_frame = None
        self.trackers = []
        self.subtractor = create_subtractor(0.5, 'MOG2')
        parent.layout.addWidget(self.video_frame)

    def init_trackers(self):
        self.timer.stop()
        winname = "Roi selection"
        rois = cv2.selectROIs(winname, img=self.raw_frame, fromCenter=False)
        cv2.destroyWindow(winname)
        for roi in rois:
            tracker = create_tracker('meanshift', roi, self.raw_frame)
            self.trackers.append(tracker)
        self.timer.start()

    def nextFrameSlot(self):
        start = cv2.getTickCount()

        # read frame
        ret, frame = self.cap.read()

        if ret:
            # scale image
            frame, width, height = scale_image(frame, 0.5)
            self.raw_frame = frame.copy()

            # operate with frame (tracking and subtraction)
            if len(self.trackers) > 0:
                for t in self.trackers:
                    t.update(self.raw_frame, frame)

            processed = self.subtractor.apply(self.raw_frame)

            draw_contours(frame, processed)

            # add stats
            frame = print_stats(frame, width, height, start, self.cap.get(cv2.CAP_PROP_POS_MSEC),
                                self.length/self.frame_rate)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            self.video_frame.setPixmap(pix)

    def start(self):
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000.0/self.frame_rate)

    def pause(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QWidget, self).deleteLater()


class VideoDisplayWidget(QWidget):
    def __init__(self, parent):
        super(VideoDisplayWidget, self).__init__(parent)
        self.layout = QFormLayout(self)
        self.startButton = QPushButton('Start', parent)
        self.startButton.clicked.connect(parent.startCapture)
        self.startButton.setFixedWidth(50)
        self.pauseButton = QPushButton('Pause', parent)
        self.pauseButton.setFixedWidth(50)
        self.layout.addRow(self.startButton, self.pauseButton)
        self.setLayout(self.layout)


class ControlWindow(QMainWindow):
    def __init__(self):
        super(ControlWindow, self).__init__()
        self.setGeometry(50, 50, 800, 600)
        self.setWindowTitle("PyTrack")

        self.capture = None

        self.isVideoFileLoaded = False

        self.quitAction = QAction("Exit", self)
        self.quitAction.setShortcut("Ctrl+Q")
        self.quitAction.triggered.connect(self.closeApplication)

        self.openVideoFile = QAction("Open Video File", self)
        self.openVideoFile.setShortcut("Ctrl+Shift+V")
        self.openVideoFile.triggered.connect(self.loadVideoFile)

        self.mainMenu = self.menuBar()
        self.fileMenu = self.mainMenu.addMenu('File')
        self.fileMenu.addAction(self.openVideoFile)
        self.fileMenu.addAction(self.quitAction)

        self.videoDisplayWidget = VideoDisplayWidget(self)
        self.setCentralWidget(self.videoDisplayWidget)

    def keyPressEvent(self, a0: QtGui.QKeyEvent):
        if a0.key() == QtCore.Qt.Key_T:
            if self.capture is not None and self.isVideoFileLoaded:
                self.capture.init_trackers()

    def startCapture(self):
        if not self.capture and self.isVideoFileLoaded:
            self.capture = VideoCapture(self.videoFileName, self.videoDisplayWidget)
            self.videoDisplayWidget.pauseButton.clicked.connect(self.capture.pause)
        self.capture.start()

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None

    def loadVideoFile(self):
        try:
            self.videoFileName = QFileDialog.getOpenFileName(self, 'Select a Video File')
            self.isVideoFileLoaded = True
        except:
            print("Please Select a Video File")

    def closeApplication(self):
        choice = QMessageBox.question(self, 'Message', 'Do you really want to exit?', QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            print("Closing....")
            sys.exit()
        else:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ControlWindow()
    window.show()
    sys.exit(app.exec_())
