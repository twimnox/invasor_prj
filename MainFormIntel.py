import MainWindow as mw_gui
import cv2
import yaml
import os
from PySide import QtCore, QtGui

ML_CLASSES = 0
TMP_DIR = "./tmp"
CONFIG_DIR = "./yaml"


def load_img(img_path):
    tmp_name = os.path.join(TMP_DIR, "_tmp_img.jpg") #use tempfile python lib?
    tmp_img = cv2.imread(img_path)
    cv2.imwrite(tmp_name, tmp_img)

    img = QtGui.QImage(tmp_name)
    ui.imglabel.setScaledContents(True)
    ui.imglabel.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
    ui.imglabel.setPixmap(QtGui.QPixmap.fromImage(img))


def import_yaml(path_to_yaml):
    global ML_CLASSES
    stream = file(os.path.join(CONFIG_DIR, path_to_yaml), 'r')
    ML_CLASSES = yaml.load(stream)


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = mw_gui.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    load_img("./odm_orthphoto.png")
    import_yaml('ml_classes.yaml')
    print ML_CLASSES
    sys.exit(app.exec_())
