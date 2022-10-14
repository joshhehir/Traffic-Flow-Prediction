from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTextEdit

from data.data import process_data
from train import main


class Ui_MainWindow(object):
    def __init__(self):
        self.output_terminal_textEdit = None
        self.process = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(603, 456)
        MainWindow.setStyleSheet("background-color: rgb(200, 184, 189);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 80, 581, 121))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(0, 40, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(0, 0, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(0, 80, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setGeometry(QtCore.QRect(110, 10, 471, 22))
        self.comboBox.setStyleSheet("background-color: rgb(255, 247, 226);")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_2.setGeometry(QtCore.QRect(110, 50, 471, 22))
        self.comboBox_2.setStyleSheet("background-color: rgb(255, 247, 226);")
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_3 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_3.setGeometry(QtCore.QRect(110, 90, 471, 22))
        self.comboBox_3.setStyleSheet("background-color: rgb(255, 247, 226);")
        self.comboBox_3.setObjectName("comboBox_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 220, 581, 23))
        self.pushButton.setStyleSheet("background-color: rgb(158, 255, 166);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.train_model)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 20, 581, 23))
        self.pushButton_2.setStyleSheet("background-color: rgb(255, 69, 87);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.load_data)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(13, 260, 581, 171))
        self.textEdit.setStyleSheet("background-color: rgb(0, 0, 0);\n"
                                    "color: rgb(0, 255, 0);")
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "TFPS - TRAIN"))
        self.label_2.setText(_translate("MainWindow", " Junction"))
        self.label.setText(_translate("MainWindow", " Model"))
        self.label_3.setText(_translate("MainWindow", " Scats"))
        self.comboBox.setItemText(0, _translate("MainWindow", "GRU"))
        self.comboBox.setItemText(1, _translate("MainWindow", "LSTM"))
        self.comboBox.setItemText(2, _translate("MainWindow", "SAE"))
        self.pushButton.setText(_translate("MainWindow", "TRAIN MODEL"))
        self.pushButton_2.setText(_translate("MainWindow", "LOAD DATA"))
        self.textEdit.setHtml(_translate("MainWindow",
                                         "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" "
                                         "\"http://www.w3.org/TR/REC-html40/strict.dtd\">\n "
                                         "<html><head><meta name=\"qrichtext\" content=\"1\" /><style "
                                         "type=\"text/css\">\n "
                                         "p, li { white-space: pre-wrap; }\n"
                                         "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; "
                                         "font-size:8.25pt; font-weight:400; "
                                         "font-style:normal;\">\n "
                                         "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; "
                                         "margin-left:0px; margin-right:0px; "
                                         "-qt-block-indent:0; text-indent:0px;\"><br "
                                         "/></p></body></html>"))

    def load_data(self):
        process_data("data/2006.csv", 12)
        self.pushButton_2.setStyleSheet("background-color: rgb(35, 255, 35);")

    def train_model(self):
        if self.comboBox.itemText(0):
            main('--gru')

        if self.comboBox.itemText(1):
            main('--lstm')

        if self.comboBox.itemText(2):
            main('--saes')


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
