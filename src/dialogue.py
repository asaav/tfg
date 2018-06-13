from PyQt5 import QtWidgets, QtCore


class HelpDialogue(QtWidgets.QDialog):
    def __init__(self, gui: bool):
        super(HelpDialogue, self).__init__()
        self.setObjectName("Dialog")
        self.setWindowTitle("Help")
        self.resize(278, 131)
        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 1, 271, 129))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.label.setMargin(10)
        self.verticalLayout.addWidget(self.label)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.verticalLayoutWidget)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(True)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.accepted.connect(self.accept)
        QtCore.QMetaObject.connectSlotsByName(self)
        if not gui:
            self.label.setText("Key controls\n"
                               "Space: Pause/resume video\n"
                               "K: Rewind 5 seconds\n"
                               "L: Forward 5 seconds\n"
                               "J: Start/stop action\n"
                               "T: Initiate trackers")
        else:
            self.label.setText("T: Initiate trackers\n"
                               "J: Start/stop action")
