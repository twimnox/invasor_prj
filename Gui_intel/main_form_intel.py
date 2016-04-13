from cPickle import load

import Gui.MainWindow as mw_gui
from dialog_load_data_intel import Ui_Data_Dialog_interaction
from dialog_import_model_intel import Ui_Import_Model_Interaction
from dialog_specifications_intel import Ui_Specifications_Interaction
import cv2
import os
from PySide import QtCore, QtGui
from Utils import variables as vars
from main_right_panel_widget_intel import ImageWidget
from Gui.main_right_panel_widget import Ui_main_right_panel_widget
from Gui.MainWindow import Ui_MainWindow
from Image.scan import Scan


def run_model_clicked():
    Img_Scanner = Scan(global_vars)
    Img_Scanner.scan_img()


if __name__ == "__main__":
    import sys

    global_vars = vars.Variables()

    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = mw_gui.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()


    img_widget = ImageWidget(MainWindow, global_vars)
    ui.splitter_right_layout.addWidget(img_widget)


    Ui_dit = Ui_Data_Dialog_interaction(global_vars)
    Ui_idm = Ui_Import_Model_Interaction(global_vars)
    Ui_ds = Ui_Specifications_Interaction(global_vars)



    #CONNECTS

    # tab actions:
    ui.actionData_Import.triggered.connect(Ui_dit.open_dialog)
    ui.actionImport_Model.triggered.connect(Ui_idm.open_dialog)
    ui.actionSpecifications.triggered.connect(Ui_ds.open_dialog)
    ui.actionRunModel.triggered.connect(run_model_clicked)


    # user interface updates:
    Ui_dit.update_img.connect(img_widget.update_image)
    Ui_idm.update_model_data.connect(Ui_ds.update_specs)


    sys.exit(app.exec_())
