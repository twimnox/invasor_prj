from cPickle import load

import Gui.MainWindow as mw_gui
from dialog_load_data_intel import Ui_Data_Dialog_interaction as Ui_Data_Import_Menu
import cv2
import yaml
import os
from PySide import QtCore, QtGui
from Utils import variables as vars
from main_right_panel_widget_intel import ImageWidget
from Gui.main_right_panel_widget import Ui_main_right_panel_widget
from Gui.MainWindow import Ui_MainWindow




# class Ui_MainWindow_interaction(object):
#
#     def __init__(self, ui, vars):
#         self.ui = ui
#         self.variables = vars
#
#     @staticmethod
#     def import_yaml(path_to_yaml):
#         global ML_CLASSES
#         stream = file(os.path.join(CONFIG_DIR, path_to_yaml), 'r')
#         return yaml.load(stream)
#
#     def load_img(self, img_path):
#         tmp_name = os.path.join(TMP_DIR, "_tmp_img.jpg") #use tempfile python lib?
#         tmp_img = cv2.imread(img_path)
#         cv2.imwrite(tmp_name, tmp_img)
#         img = QtGui.QImage(tmp_name)
#
#         ui.imglabel.setScaledContents(True)
#         ui.imglabel.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
#         display_img = QtGui.QPixmap.fromImage(img)
#         ui.imglabel.setPixmap(display_img)
#
#
#     @QtCore.Slot()
#     def on_update_image(self):
#         img_path = self.variables.import_data_path
#         self.load_img(img_path)



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

    Ui_dit = Ui_Data_Import_Menu(global_vars)


    #CONNECTS
    ui.actionData_Import.triggered.connect(Ui_dit.open_dialog)
    Ui_dit.update_img.connect(img_widget.update_image)


    sys.exit(app.exec_())
