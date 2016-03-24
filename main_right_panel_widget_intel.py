from PySide.QtCore import QObject, Signal, Slot
from PySide import QtCore, QtGui
import Gui.MainWindow as main_ui
import cv2
import os
from Gui.main_right_panel_widget import Ui_main_right_panel_widget



class ImageWidget(QtGui.QWidget):
    def __init__(self, parent, variables):
        super(ImageWidget, self).__init__()

        # RightPanel = QtGui.QWidget
        self.ui = Ui_main_right_panel_widget()
        self.ui.setupUi(self)

        self.variables = variables



    #SIGNALS


    #SLOTS

    @QtCore.Slot()
    def update_image(self):
        img_path = self.variables.import_data_path
        self.load_img(img_path)


    def load_img(self, img_path):
        tmp_name = os.path.join(self.variables.TMP_DIR, "_tmp_img.jpg") #use tempfile python lib?
        tmp_img = cv2.imread(img_path)
        cv2.imwrite(tmp_name, tmp_img)
        img = QtGui.QImage(tmp_name)
        display_img = QtGui.QPixmap.fromImage(img)
        # self.ui.imglabel.setPixmap(display_img)


        scene = QtGui.QGraphicsScene()
        pixmap = QtGui.QPixmap(tmp_name)
        pixmap = pixmap.scaled(222, 222, aspectRatioMode=QtCore.Qt.KeepAspectRatio,
                       transformMode=QtCore.Qt.SmoothTransformation)
        pixItem = QtGui.QGraphicsPixmapItem(pixmap)
        scene.addPixmap(pixmap)
        self.ui.graphicsView.setScene(scene)






