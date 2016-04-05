import Gui.dialog_model as dm
from PySide import QtCore, QtGui
from PySide.QtCore import QObject, Signal, Slot
import unicodedata


class Ui_Import_Model_Interaction(QObject):

    def __init__(self, variables):
        QObject.__init__(self)

        self.variables = variables
        self.DataDialog = QtGui.QDialog()
        self.ui = dm.Ui_Dialog()
        self.ui.setupUi(self.DataDialog)

        #SLOTS
        self.ui.btn_model_data_folder.clicked.connect(self.open_file_dialog)
        self.ui.btn_cancel.clicked.connect(self.cancel_click)
        self.ui.btn_ok.clicked.connect(self.ok_click)



    #SIGNALS
    update_model_data = Signal() #Signal(String), string = "campo1#campo2#campo3..."



    def cancel_click(self):
        self.DataDialog.close()

    def ok_click(self):
        self.update_model_data.emit()
        self.DataDialog.close()


    def open_dialog(self):
        """
        Opens the Load Data dialog
        """
        self.DataDialog.show()


    def open_file_dialog(self):
        """
        Opens file browser prompt

        """
        fileDialog = QtGui.QFileDialog()
        # fileDialog.setNameFilters("Image files (*.png *.tif, *.jpg)")
        fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
        filename = (fileDialog.getExistingDirectory())

        format_filename = unicodedata.normalize('NFKD', filename).encode('ascii','ignore')

        self.variables.export_data_path = format_filename
        self.ui.text_model_data_folder.setText(format_filename)








