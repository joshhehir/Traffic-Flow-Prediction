import os
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from data.scats import ScatsData

SCATS_DATA = ScatsData()

default_font = QtGui.QFont()
default_font.setFamily("Arial")
default_font.setPointSize(10)

label_font = QtGui.QFont()
label_font.setFamily("Arial")
label_font.setPointSize(10)
label_font.setBold(True)
label_font.setWeight(75)


class ConsoleStream(QtCore.QObject):
    """Handles the stdout functions """
    text_output = QtCore.pyqtSignal(str)

    def write(self, text):
        """Writes the console output to the text_edit widget"""
        self.text_output.emit(str(text))

    def flush(self):
        """This function is here to avoid compiler errors"""
        pass


class UiRouting(object):

    def __init__(self, main):
        self.threads = []
        self.scats_info = {}

        # Initialise widgets
        self.main = main
        self.main_widget = QtWidgets.QWidget(main)
        self.predict_push_button = QtWidgets.QPushButton(self.main_widget)
        self.origin_scats_number_combo_box = QtWidgets.QComboBox(self.main_widget)
        self.origin_scats_number_label = QtWidgets.QLabel(self.main_widget)
        self.destination_scats_number_combo_box = QtWidgets.QComboBox(self.main_widget)
        self.destination_scats_number_label = QtWidgets.QLabel(self.main_widget)
        self.date_input = QtWidgets.QDateTimeEdit(self.main_widget)
        self.date_input_label = QtWidgets.QLabel(self.main_widget)
        self.settings_layout = QtWidgets.QFormLayout()
        self.vertical_layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.date_time_layout = QtWidgets.QFormLayout()
        self.out_textEdit = QtWidgets.QPlainTextEdit(self.main_widget)

        sys.stdout = ConsoleStream(text_output=self.display_output)

        self.main.setObjectName("main_window")
        self.main_widget.setObjectName("main_widget")
        self.vertical_layout.setObjectName("vertical_layout")
        self.settings_layout.setObjectName("settings_layout")
        self.date_input.setObjectName("date_input")
        self.date_input_label.setObjectName("date_input_label")
        self.origin_scats_number_label.setObjectName("origin_scats_number_label")
        self.origin_scats_number_combo_box.setObjectName("origin_scats_number_combo_box")
        self.destination_scats_number_label.setObjectName("destination_scats_number_label")
        self.destination_scats_number_combo_box.setObjectName("destination_scats_number_combo_box")
        self.predict_push_button.setObjectName("predict_push_button")
        self.date_time_layout.setObjectName("date_time_layout")
        self.out_textEdit.setReadOnly(True)
        self.out_textEdit.setObjectName("out_textEdit")

    def __del__(self):
        for thread in self.threads:
            thread.join()

    def display_output(self, text):
        """Displays the terminal text in the out_textEdit widget"""
        cursor = self.out_textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.out_textEdit.setTextCursor(cursor)
        self.out_textEdit.ensureCursorVisible()

    def setup(self):
        """Sets up the layout and widgets to be used in the GUI"""
        self.set_text(main_window)
        self.set_style(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)
        self.init_layouts()
        self.init_widgets()

    def init_layouts(self):
        """Creates the layouts for the widgets"""
        # Main window setup
        main_window.setCentralWidget(self.main_widget)
        self.main.resize(800, 450)

        # Vertical layout setup
        self.vertical_layout.addLayout(self.settings_layout)
        self.vertical_layout.addLayout(self.date_time_layout)
        self.vertical_layout.addWidget(self.predict_push_button)
        self.vertical_layout.addWidget(self.out_textEdit)

        self.date_time_layout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.date_input_label)
        self.date_time_layout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.date_input)

        # Settings layout setup
        self.settings_layout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.origin_scats_number_label)
        self.settings_layout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.origin_scats_number_combo_box)
        self.settings_layout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.destination_scats_number_label)
        self.settings_layout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.destination_scats_number_combo_box)

    def set_text(self, main):
        """Sets the text for all the buttons and labels"""
        translate = QtCore.QCoreApplication.translate
        main.setWindowTitle(translate("main_window", "Traffic Flow Prediction System - Routing Program"))

        self.origin_scats_number_label.setFont(default_font)
        self.origin_scats_number_combo_box.setFont(default_font)
        self.destination_scats_number_label.setFont(default_font)
        self.destination_scats_number_combo_box.setFont(default_font)
        self.date_input.setFont(default_font)
        self.date_input_label.setFont(default_font)
        self.predict_push_button.setFont(default_font)
        self.out_textEdit.setFont(default_font)

        translate = QtCore.QCoreApplication.translate

        self.origin_scats_number_label.setText(translate("main_window", "Origin Scats Number"))
        self.destination_scats_number_label.setText(translate("main_window", "Destination Scats Number"))
        self.date_input_label.setText(translate("main_window", "Date/Time"))
        self.predict_push_button.setText(translate("main_window", "Predict Route"))

    def set_style(self, main):
        """Sets the style of the GUI, the colors, etc..."""
        main.setStyleSheet("background-color: rgb(140, 140, 140);")
        self.origin_scats_number_combo_box.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.destination_scats_number_combo_box.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.date_input.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.predict_push_button.setStyleSheet("background-color: rgb(158, 255, 166);")
        self.out_textEdit.setStyleSheet("background-color: rgb(0, 0, 0);\n" "color: rgb(0, 255, 0);")

    def init_widgets(self):
        """Initialises the widgets to be used"""
        _translate = QtCore.QCoreApplication.translate
        scats_numbers = SCATS_DATA.get_all_scats_numbers()

        self.origin_scats_number_combo_box.addItem("")
        self.destination_scats_number_combo_box.addItem("")
        for scats in scats_numbers:
            self.origin_scats_number_combo_box.addItem(str(scats))
            self.destination_scats_number_combo_box.addItem(str(scats))

        self.predict_push_button.setEnabled(False)

        # Adds functionality to the controls
        # self.predict_push_button.clicked.connect(self.predict)
        self.origin_scats_number_combo_box.currentIndexChanged.connect(self.scats_number_changed)
        self.destination_scats_number_combo_box.currentIndexChanged.connect(self.scats_number_changed)

    def scats_number_changed(self):
        """Updates the combo boxes when the scats site is changed"""
        origin_index = self.origin_scats_number_combo_box.currentIndex()
        origin_value = self.origin_scats_number_combo_box.itemText(origin_index)

        destination_index = self.origin_scats_number_combo_box.currentIndex()
        destination_value = self.origin_scats_number_combo_box.itemText(destination_index)

        self.element_changed()

    def element_changed(self):
        """Updates the combo boxes and enables the predict route button"""
        origin_scats_combo_value = self.origin_scats_number_combo_box.itemText(
            self.origin_scats_number_combo_box.currentIndex())

        destination_scats_combo_value = self.destination_scats_number_combo_box.itemText(
            self.destination_scats_number_combo_box.currentIndex())

        self.predict_push_button.setEnabled(
            origin_scats_combo_value != "" and destination_scats_combo_value != "")


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = UiRouting(main_window)
    ui.setup()
    main_window.show()
    sys.exit(app.exec_())
