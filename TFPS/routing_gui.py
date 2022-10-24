import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from data.scats import ScatsData
from application import get_graph

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


class UiRouting(object):

    def __init__(self, main):
        self.threads = []
        self.scats_info = {}

        # Initialise widgets
        self.main = main
        self.main_widget = QtWidgets.QWidget(main)
        self.predict_push_button = QtWidgets.QPushButton(self.main_widget)
        self.model_comboBox = QtWidgets.QComboBox(self.main_widget)
        self.model_label = QtWidgets.QLabel(self.main_widget)
        self.origin_scats_number_combo_box = QtWidgets.QComboBox(self.main_widget)
        self.origin_scats_number_label = QtWidgets.QLabel(self.main_widget)
        self.destination_scats_number_combo_box = QtWidgets.QComboBox(self.main_widget)
        self.destination_scats_number_label = QtWidgets.QLabel(self.main_widget)
        self.time_input = QtWidgets.QTimeEdit(self.main_widget)
        self.time_input_label = QtWidgets.QLabel(self.main_widget)
        self.settings_layout = QtWidgets.QFormLayout()
        self.vertical_layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.time_layout = QtWidgets.QFormLayout()
        self.out_textEdit = QtWidgets.QPlainTextEdit(self.main_widget)

        sys.stdout = ConsoleStream(text_output=self.display_output)

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
        default_font = QtGui.QFont()
        default_font.setFamily("Arial")
        default_font.setPointSize(10)

        label_font = QtGui.QFont()
        label_font.setFamily("Arial")
        label_font.setPointSize(10)
        label_font.setBold(True)
        label_font.setWeight(75)

        # Main window setup
        self.main.setObjectName("main_window")
        self.main.resize(1200, 800)
        self.main_widget.setObjectName("main_widget")

        self.vertical_layout.setObjectName("vertical_layout")
        self.settings_layout.setObjectName("settings_layout")

        self.model_label.setFont(default_font)
        self.model_label.setObjectName("model_label")
        self.settings_layout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.model_label)

        self.model_comboBox.setFont(default_font)
        self.model_comboBox.setObjectName("model_comboBox")
        self.settings_layout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.model_comboBox)

        self.origin_scats_number_label.setFont(default_font)
        self.origin_scats_number_label.setObjectName("origin_scats_number_label")
        self.settings_layout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.origin_scats_number_label)

        self.origin_scats_number_combo_box.setFont(default_font)
        self.origin_scats_number_combo_box.setObjectName("origin_scats_number_combo_box")
        self.settings_layout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.origin_scats_number_combo_box)

        self.destination_scats_number_label.setFont(default_font)
        self.destination_scats_number_label.setObjectName("destination_scats_number_label")
        self.settings_layout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.destination_scats_number_label)

        self.destination_scats_number_combo_box.setFont(default_font)
        self.destination_scats_number_combo_box.setObjectName("destination_scats_number_combo_box")
        self.settings_layout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.destination_scats_number_combo_box)

        self.time_input.setFont(default_font)
        self.time_input.setObjectName("time_input")
        self.time_layout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.time_input)

        self.time_input_label.setFont(default_font)
        self.time_input_label.setObjectName("time_input_label")
        self.time_layout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.time_input_label)

        self.predict_push_button.setFont(default_font)
        self.predict_push_button.setObjectName("predict_push_button")

        self.out_textEdit.setFont(default_font)
        self.out_textEdit.setReadOnly(True)
        self.out_textEdit.setObjectName("out_textEdit")

        self.time_layout.setObjectName("time_layout")

        # Vertical layout setup
        self.vertical_layout.addLayout(self.settings_layout)
        self.vertical_layout.addLayout(self.time_layout)
        self.vertical_layout.addWidget(self.predict_push_button)
        self.vertical_layout.addWidget(self.out_textEdit)

        main_window.setCentralWidget(self.main_widget)

        self.set_text(main_window)
        self.set_style(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)
        self.init_widgets()

    def set_text(self, main):
        """Sets the text for all the buttons and labels"""
        translate = QtCore.QCoreApplication.translate
        main.setWindowTitle(translate("main_window", "Traffic Flow Prediction System - Routing Program"))
        self.model_label.setText(translate("main_window", "Model"))
        self.origin_scats_number_label.setText(translate("main_window", "Origin Scats Number"))
        self.destination_scats_number_label.setText(translate("main_window", "Destination Scats Number"))
        self.time_input_label.setText(translate("main_window", "Time"))
        self.predict_push_button.setText(translate("main_window", "Predict Route"))

    def set_style(self, main):
        """Sets the style of the GUI, the colors, etc..."""

        main.setStyleSheet("background-color: rgb(140, 140, 140);")
        self.model_comboBox.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.origin_scats_number_combo_box.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.destination_scats_number_combo_box.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.time_input.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.predict_push_button.setStyleSheet("background-color: rgb(158, 255, 166);")
        self.out_textEdit.setStyleSheet("background-color: rgb(0, 0, 0);\n" "color: rgb(0, 255, 0);")

    def init_widgets(self):
        """Initialises the widgets to be used"""
        _translate = QtCore.QCoreApplication.translate
        scats_numbers = SCATS_DATA.get_all_scats_numbers()

        models = ["LSTM", "GRU", "SAEs", "SRNN"]
        for model in models:
            self.model_comboBox.addItem(model)

        self.origin_scats_number_combo_box.addItem("")
        self.destination_scats_number_combo_box.addItem("")
        for scats in scats_numbers:
            self.origin_scats_number_combo_box.addItem(str(scats))
            self.destination_scats_number_combo_box.addItem(str(scats))

        self.predict_push_button.setEnabled(False)

        # Adds functionality to the controls
        self.predict_push_button.clicked.connect(self.route_process)
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
        model_combo_value = self.model_comboBox.itemText(self.model_comboBox.currentIndex()).lower()

        origin_scats_combo_value = self.origin_scats_number_combo_box.itemText(
            self.origin_scats_number_combo_box.currentIndex())

        destination_scats_combo_value = self.destination_scats_number_combo_box.itemText(
            self.destination_scats_number_combo_box.currentIndex())

        self.predict_push_button.setEnabled(
            model_combo_value != "" and origin_scats_combo_value != "" and destination_scats_combo_value != "")

    def route(self):
        """Passes routing parameters"""
        model_combo_value = self.model_comboBox.itemText(self.model_comboBox.currentIndex()).lower()

        origin_scats_number = self.origin_scats_number_combo_box.itemText(self.origin_scats_number_combo_box.currentIndex())
        if origin_scats_number != "":
            origin_scats_number = int(origin_scats_number)

        destination_scats_number = self.destination_scats_number_combo_box.itemText(self.destination_scats_number_combo_box.currentIndex())
        if destination_scats_number != "":
            destination_scats_number = int(destination_scats_number)

        time_input_value = self.time_input.time()
        routes = 5
        graph = get_graph()
        graph.get_paths(origin_scats_number, destination_scats_number, routes, model_combo_value, time_input_value)

    def route_process(self):
        """Enables threads for the training GUI"""
        training_threads = []
        t = threading.Thread(target=self.route)
        training_threads.append(t)
        self.threads.append(t)

        for thread in training_threads:
            thread.start()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = UiRouting(main_window)
    ui.setup()
    main_window.show()
    sys.exit(app.exec_())
