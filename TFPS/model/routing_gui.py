import os
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from data.scats import ScatsData

SCATS_DATA = ScatsData()


class Ui_Routing(object):

    def __init__(self, main):
        # Initialise widgets
        self.main = main
        self.main_widget = QtWidgets.QWidget(main)
        self.predict_push_button = QtWidgets.QPushButton(self.main_widget)
        self.origin_junction_combo_box = QtWidgets.QComboBox(self.main_widget)
        self.origin_junction_label = QtWidgets.QLabel(self.main_widget)
        self.origin_scats_number_combo_box = QtWidgets.QComboBox(self.main_widget)
        self.origin_scats_number_label = QtWidgets.QLabel(self.main_widget)
        self.date_input = QtWidgets.QDateTimeEdit(self.main_widget)
        self.date_input_label = QtWidgets.QLabel(self.main_widget)
        self.settings_layout = QtWidgets.QFormLayout()
        self.vertical_layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.date_time_layout = QtWidgets.QFormLayout()
        self.text_output = QtWidgets.QLabel()
