from PySide.QtCore import QObject, Signal, Slot
from PySide import QtCore, QtGui
import Gui.MainWindow as main_ui
import cv2
import os
from Gui.main_right_panel_widget import Ui_main_right_panel_widget



class ImageWidget(QtGui.QWidget):
    def __init__(self, parent, variables):
        super(ImageWidget, self).__init__()
        self.ui = Ui_main_right_panel_widget()
        self.ui.setupUi(self)
        self.variables = variables

        #SLOTS
        self.ui.btn_zoom_in.clicked.connect(self.zoom_in)
        self.ui.btn_zoom_out.clicked.connect(self.zoom_out)




    #SIGNALS


    #SLOTS

    @QtCore.Slot()
    def update_image(self):
        img_path = self.variables.import_data_path
        self.load_img(img_path)

    @QtCore.Slot()
    def zoom_in(self):
        scale = self.variables.ZOOM_FACTOR*(1+0.25)
        self.variables.ZOOM_FACTOR = scale
        self.zoom_operation()
        return

    @QtCore.Slot()
    def zoom_out(self):
        scale = self.variables.ZOOM_FACTOR*(1-0.25)
        self.variables.ZOOM_FACTOR = scale
        self.zoom_operation()
        return


    def zoom_operation(self):
        if self.variables.import_data_path != "empty":
            img_path = self.variables.import_data_path
            scale = self.variables.ZOOM_FACTOR

            self.ui.graphicsView.scene()

            tmp_name = os.path.join(self.variables.TMP_DIR, "_tmp_img.jpg") #use tempfile python lib?
            tmp_img = cv2.imread(img_path)
            curr_height, curr_width = tmp_img.shape[:2]
            cv2.imwrite(tmp_name, tmp_img)
            img = QtGui.QImage(tmp_name)
            display_img = QtGui.QPixmap.fromImage(img)


            scene = QtGui.QGraphicsScene()
            pixmap = QtGui.QPixmap(tmp_name)
            pixmap = pixmap.scaled(curr_width*scale, curr_height*scale, aspectRatioMode=QtCore.Qt.KeepAspectRatio,
                                   transformMode=QtCore.Qt.SmoothTransformation)
            pixItem = QtGui.QGraphicsPixmapItem(pixmap)
            scene.addPixmap(pixmap)
            self.ui.graphicsView.setScene(scene)



    def load_img(self, img_path):
        """

        :param img_path: path to image to openj
        """

        #TODO: save image files with multiple sizes

        tmp_name = os.path.join(self.variables.TMP_DIR, "_tmp_img.jpg") #use tempfile python lib?
        tmp_img = cv2.imread(img_path)
        cv2.imwrite(tmp_name, tmp_img)
        img = QtGui.QImage(tmp_name)
        display_img = QtGui.QPixmap.fromImage(img)

        scene = QtGui.QGraphicsScene()
        pixmap = QtGui.QPixmap(tmp_name)
        pixItem = QtGui.QGraphicsPixmapItem(pixmap)
        scene.addPixmap(pixmap)
        self.ui.graphicsView.setScene(scene)






