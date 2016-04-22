from PySide.QtCore import QObject, Signal, Slot
from PySide import QtCore, QtGui
import Gui.MainWindow as main_ui
import cv2
import os
from Gui.main_left_panel_widget import Ui_main_left_panel_widget
from Image.layers import Layers



class LayersWidget(QtGui.QWidget):
    def __init__(self, parent, variables):
        super(LayersWidget, self).__init__()
        self.ui = Ui_main_left_panel_widget()
        self.ui.setupUi(self)
        self.variables = variables


    @QtCore.Slot()
    def clear_check_boxes(self):
        self.ui.label_nothing_to_show.setParent(None)
        for i in range(len(self.variables.classes)):
            self.variables.layers_checkboxes[i].setParent(None)



    @QtCore.Slot()
    def add_classes_checkboxes(self):
        self.ui.label_nothing_to_show.setParent(None)

        lyr = Layers(self.variables)

        # Add "All" checkbox option
        c = QtGui.QCheckBox("All")
        self.ui.formLayout_2.addWidget(c)
        self.variables.layers_checkboxes.append(c)

        for i in range(len(self.variables.classes)):
            c = QtGui.QCheckBox(self.variables.classes[i])
            self.ui.formLayout_2.addWidget(c)
            self.variables.layers_checkboxes.append(c)








