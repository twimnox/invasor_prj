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

        self.scaleFactor = 0.0

        self.ui.imglabel.setBackgroundRole(QtGui.QPalette.Base)
        self.ui.imglabel.setSizePolicy(QtGui.QSizePolicy.Ignored,
                                      QtGui.QSizePolicy.Ignored)
        self.ui.imglabel.setScaledContents(True)
        self.ui.img_scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
        self.createActions()



    #SIGNALS


    #SLOTS

    @QtCore.Slot()
    def update_image(self):
        img_path = self.variables.import_data_path
        self.load_img(img_path)


    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()


    def createActions(self):

        self.ui.zoomInAct = QtGui.QAction("Zoom &In (25%)", self,
                                       shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QtGui.QAction("Zoom &Out (25%)", self,
                                        shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QtGui.QAction("&Normal Size", self,
                                           shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QtGui.QAction("&Fit to Window", self,
                                            enabled=False, checkable=True, shortcut="Ctrl+F",
                                            triggered=self.fitToWindow)


    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(selfscaleFactor * self.ui.imglabel.pixmap().size())

        self.adjustScrollBar(self.ui.img_scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.ui.img_scrollArea.verticalScrollBar(), factor)

        # self.ui.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        # self.ui.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep()/2)))





    def load_img(self, img_path):
        tmp_name = os.path.join(self.variables.TMP_DIR, "_tmp_img.jpg") #use tempfile python lib?
        tmp_img = cv2.imread(img_path)
        cv2.imwrite(tmp_name, tmp_img)
        img = QtGui.QImage(tmp_name)
        display_img = QtGui.QPixmap.fromImage(img)
        self.ui.imglabel.setPixmap(display_img)
