import threading

from PyQt5 import QtCore, QtGui, QtWidgets
from data.scats import ScatsData
from train import train_with_args

SCATS_DATA = ScatsData()


class ConsoleStream(QtCore.QObject):
    """Handles the stdout functions """
    text_output = QtCore.pyqtSignal(str)

    def write(self, text):
        """Writes the console output to the text_edit widget"""
        self.text_output.emit(str(text))

    def flush(self):
        """This function is here to avoid compiler errors"""
        pass


class UiTrain(object):
    """The GUI for training the models."""

    def __init__(self, main):
        self.threads = []
        self.scats_info = {}

        # Initialise widgets
        self.main = main
        self.main_widget = QtWidgets.QWidget(main)
        self.out_textEdit = QtWidgets.QPlainTextEdit(self.main_widget)
        self.train_Button = QtWidgets.QPushButton(self.main_widget)
        self.junction_comboBox = QtWidgets.QComboBox(self.main_widget)
        self.junction_label = QtWidgets.QLabel(self.main_widget)
        self.scats_comboBox = QtWidgets.QComboBox(self.main_widget)
        self.scats_label = QtWidgets.QLabel(self.main_widget)
        self.model_comboBox = QtWidgets.QComboBox(self.main_widget)
        self.model_label = QtWidgets.QLabel(self.main_widget)
        self.settings_layout = QtWidgets.QFormLayout()
        self.vertical_layout = QtWidgets.QVBoxLayout(self.main_widget)

        sys.stdout = ConsoleStream(text_output=self.display_output)

    def __del__(self):
        """Resets the console if the GUI is closed"""
        sys.stdout = sys.__stdout__

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
        default_font = QtGui.QFont()
        default_font.setFamily("Arial")
        default_font.setPointSize(10)

        label_font = QtGui.QFont()
        label_font.setFamily("Arial")
        label_font.setPointSize(10)
        label_font.setBold(True)
        label_font.setWeight(75)

        self.main.setObjectName("main_window")
        self.main.resize(800, 450)
        self.main_widget.setObjectName("main_widget")

        self.vertical_layout.setObjectName("vertical_layout")
        self.settings_layout.setFormAlignment(QtCore.Qt.AlignCenter)
        self.settings_layout.setObjectName("settings_layout")

        self.model_label.setFont(default_font)
        self.model_label.setObjectName("model_label")
        self.settings_layout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.model_label)

        self.model_comboBox.setFont(default_font)
        self.model_comboBox.setObjectName("model_comboBox")
        self.settings_layout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.model_comboBox)

        self.scats_label.setFont(default_font)
        self.scats_label.setObjectName("scats_label")
        self.settings_layout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.scats_label)

        self.scats_comboBox.setFont(default_font)
        self.scats_comboBox.setObjectName("scats_comboBox")
        self.settings_layout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.scats_comboBox)

        self.junction_label.setFont(default_font)
        self.junction_label.setObjectName("junction_label")
        self.settings_layout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.junction_label)

        self.junction_comboBox.setFont(default_font)
        self.junction_comboBox.setObjectName("junction_comboBox")
        self.settings_layout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.junction_comboBox)

        self.vertical_layout.addLayout(self.settings_layout)

        self.train_Button.setFont(default_font)
        self.train_Button.setObjectName("train_Button")
        self.vertical_layout.addWidget(self.train_Button)

        self.out_textEdit.setReadOnly(True)
        self.out_textEdit.setObjectName("out_textEdit")

        self.out_textEdit.setFont(default_font)
        self.vertical_layout.addWidget(self.out_textEdit)
        main_window.setCentralWidget(self.main_widget)

        self.set_text(main_window)
        self.set_style(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)

        self.init_widgets()

    def set_text(self, main):
        """Sets the text for all the buttons and labels"""
        translate = QtCore.QCoreApplication.translate
        main.setWindowTitle(translate("main_window", "Traffic Flow Prediction System - Model Training Program"))
        self.model_label.setText(translate("main_window", "Training Model"))
        self.scats_label.setText(translate("main_window", "Scats Site"))
        self.junction_label.setText(translate("main_window", "Junction"))
        self.train_Button.setText(translate("main_window", "Train Model"))

    def set_style(self, main):
        """Sets the style of the GUI, the colors, etc..."""
        main.setStyleSheet("background-color: rgb(140, 140, 140);")
        self.model_comboBox.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.scats_comboBox.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.junction_comboBox.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.train_Button.setStyleSheet("background-color: rgb(158, 255, 166);")
        self.out_textEdit.setStyleSheet("background-color: rgb(0, 0, 0);\n" "color: rgb(0, 255, 0);")

    def scats_number_changed(self):
        """Updates the junction_comboBox when the scats site is changed"""
        index = self.scats_comboBox.currentIndex()
        value = self.scats_comboBox.itemText(index)

        if value == "All" or value == "":
            self.junction_comboBox.setCurrentIndex(0)
            self.junction_comboBox.setEnabled(False)
        elif value == "None":
            self.scats_comboBox.setEnabled(False)
            self.junction_comboBox.setEnabled(False)
        else:
            self.junction_comboBox.clear()

            self.junction_comboBox.addItem("All")
            for junction in self.scats_info[value]:
                self.junction_comboBox.addItem(str(junction))

            self.junction_comboBox.setEnabled(True)

    def train(self):
        """Passes training parameters"""
        scats_number = self.scats_comboBox.itemText(self.scats_comboBox.currentIndex())
        if scats_number != "All":
            scats_number = int(scats_number)
        junction = self.junction_comboBox.itemText(self.junction_comboBox.currentIndex())
        if junction != "All":
            junction = int(SCATS_DATA.get_location_id(junction))
        model = self.model_comboBox.itemText(self.model_comboBox.currentIndex()).lower()

        train_with_args(scats_number, junction, model)

    def train_process(self):
        """Enables threads for the training GUI"""
        training_threads = []
        t = threading.Thread(target=self.train)
        training_threads.append(t)
        self.threads.append(t)

        for thread in training_threads:
            thread.start()

    def init_widgets(self):
        """Initialises the widgets to be used"""
        _translate = QtCore.QCoreApplication.translate

        models = ["LSTM", "GRU", "SAEs"]
        for model in models:
            self.model_comboBox.addItem(model)

        scats_numbers = SCATS_DATA.get_all_scats_numbers()

        self.scats_comboBox.addItem("All")
        self.junction_comboBox.addItem("All")
        for scats in scats_numbers:
            self.scats_comboBox.addItem(str(scats))
            self.scats_info[str(scats)] = SCATS_DATA.get_scats_approaches(scats)

            i = 0
            for location in self.scats_info[str(scats)]:
                self.scats_info[str(scats)][i] = SCATS_DATA.get_location_name(scats, location)
                i += 1

        self.junction_comboBox.setEnabled(False)

        self.train_Button.clicked.connect(self.train_process)
        self.scats_comboBox.currentIndexChanged.connect(self.scats_number_changed)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = UiTrain(main_window)
    ui.setup()
    main_window.show()
    sys.exit(app.exec_())
