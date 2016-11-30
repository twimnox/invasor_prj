import Gui.dialog_load_data as dld
from PySide import QtCore, QtGui
from PySide.QtCore import QObject, Signal, Slot
import unicodedata
import cv2


class Ui_Data_Dialog_interaction(QObject):

    def __init__(self, variables):
        QObject.__init__(self)

        self.variables = variables
        self.DataDialog = QtGui.QDialog()
        self.ui = dld.Ui_Dialog()
        self.ui.setupUi(self.DataDialog)

        #SLOTS
        self.ui.btn_data_import_path.clicked.connect(self.open_file_dialog_import)
        self.ui.btn_data_output_folder.clicked.connect(self.open_file_dialog_export)
        self.ui.btn_cancel.clicked.connect(self.cancel_click)
        self.ui.btn_ok.clicked.connect(self.ok_click)



    #SIGNALS
    update_img = Signal()



    def cancel_click(self):
        self.DataDialog.close()

    def ok_click(self):
        self.update_img.emit()
        self.DataDialog.close()


    def open_dialog(self):
        """
        Opens the Load Data dialog
        """
        self.DataDialog.show()

    def open_file_dialog_import(self):
        """
        Opens file browser prompt

        """
        data_type = self.ui.comboBox_data_format.currentIndex()
        fileDialog = QtGui.QFileDialog()
        #fileDialog.setNameFilters("Image files (*.png *.tif, *.jpg)")
        #fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)

        if data_type == 1: #multiple images
            filename = (fileDialog.getOpenFileNames())
        else:
            filename = (fileDialog.getOpenFileName())

        format_filename = unicodedata.normalize('NFKD', filename[0]).encode('ascii','ignore')

        img = cv2.imread(format_filename)
        height, width = img.shape[:2]
        self.variables.IMG_WIDTH = width
        self.variables.IMG_HEIGTH = height

        self.variables.import_data_path = format_filename
        self.ui.text_data_import_path.setText(format_filename)

    def open_file_dialog_export(self):
        """
        Opens file browser prompt

        """
        fileDialog = QtGui.QFileDialog()
        fileDialog.setNameFilters("Image files (*.png *.tif, *.jpg)")
        fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
        filename = (fileDialog.getExistingDirectory())

        format_filename = unicodedata.normalize('NFKD', filename).encode('ascii','ignore')

        self.variables.export_data_path = format_filename
        self.ui.text_data_output_folder.setText(format_filename)








