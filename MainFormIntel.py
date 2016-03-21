import Gui.MainWindow as mw_gui
from DialogLoadDataIntel import Ui_Data_Dialog_interaction as Ui_Data_Import_Tab
import cv2
import yaml
import os
from PySide import QtCore, QtGui
from Utils import Variables as vars

from Gui.MainWindow import Ui_MainWindow

ML_CLASSES = 0
TMP_DIR = "./tmp"
CONFIG_DIR = "./yaml"


class Ui_MainWindow_interaction(object):

    def __init__(self, ui, vars):
        self.ui = ui
        self.variables = vars

    def load_img(img_path):
        tmp_name = os.path.join(TMP_DIR, "_tmp_img.jpg") #use tempfile python lib?
        tmp_img = cv2.imread(img_path)
        cv2.imwrite(tmp_name, tmp_img)

        img = QtGui.QImage(tmp_name)
        ui.imglabel.setScaledContents(True)
        ui.imglabel.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        ui.imglabel.setPixmap(QtGui.QPixmap.fromImage(img))

    @staticmethod
    def import_yaml(path_to_yaml):
        global ML_CLASSES
        stream = file(os.path.join(CONFIG_DIR, path_to_yaml), 'r')
        return yaml.load(stream)



    @QtCore.Slot()
    def on_update_image(self, int):
        img_path = self.variables.import_data_path
        #if nao for .jpg...
        tmp_name = os.path.join(TMP_DIR, "_tmp_img.jpg")  # use tempfile python lib?
        tmp_img = cv2.imread(img_path)
        cv2.imwrite(tmp_name, tmp_img)

        img = QtGui.QImage(tmp_name)
        ui.imglabel.setScaledContents(True)
        ui.imglabel.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        ui.imglabel.setPixmap(QtGui.QPixmap.fromImage(img))

class Communicate(QtCore.QObject):
    # create a new signal on the fly and name it 'speak'
    speak = QtCore.Signal(str)

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = mw_gui.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    # load_img("./odm_orthphoto.png")

    Ui_interaction = Ui_MainWindow_interaction(ui, vars.Variables())
    Ui_dit = Ui_Data_Import_Tab(vars.Variables(), Ui_MainWindow)


    ml_classes = Ui_interaction.import_yaml('ml_classes.yaml')
    print ml_classes


    #Slots
    ui.actionData_Import.triggered.connect(Ui_dit.open_dialog)
    #Ui_dit.ok_click().connect(Ui_interaction.on_update_image)
    Ui_dit.update_img.connect(Ui_interaction.on_update_image)






    sys.exit(app.exec_())
