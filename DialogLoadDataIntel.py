import Gui.dialog_load_data as dld
from PySide import QtCore, QtGui
from PySide.QtCore import QObject, Signal, Slot


class Ui_Data_Dialog_interaction(QObject):

    # signals
    update_img = Signal()

    def __init__(self, variables, main_ui):
        QObject.__init__(self)

        self.variables = variables
        #print "ola", self.variables.export_data_path
        self.DataDialog = QtGui.QDialog()
        self.ui = dld.Ui_Dialog()
        self.ui.setupUi(self.DataDialog)
        self.main_ui = main_ui

        #Signals&Slots
        self.ui.btn_data_import_path.clicked.connect(self.open_file_dialog_import)
        self.ui.btn_data_output_folder.clicked.connect(self.open_file_dialog_export)
        self.ui.btn_cancel.clicked.connect(self.on_cancel_click)
        self.ui.btn_ok.clicked.connect(self.ok_click)



    def on_cancel_click(self):
        self.DataDialog.close()

    def ok_click(self):
        self.update_img.emit()




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
        fileDialog.setNameFilters("Image files (*.png *.tif, *.jpg)")
        fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)

        if data_type == 1: #multiple images
            filename = str(fileDialog.getOpenFileNames()).encode('utf-8')
        else:
            filename = str(fileDialog.getOpenFileName()).encode('utf-8')

        self.variables.import_data_path = filename
        self.ui.text_data_import_path.setText(filename)

    def open_file_dialog_export(self):
        """
        Opens file browser prompt

        """
        fileDialog = QtGui.QFileDialog()
        fileDialog.setNameFilters("Image files (*.png *.tif, *.jpg)")
        fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
        filename = str(fileDialog.getExistingDirectory()).encode('utf-8')

        self.variables.export_data_path = filename
        self.ui.text_data_output_folder.setText(filename)








