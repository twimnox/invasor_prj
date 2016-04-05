import Gui.dialog_model as dm
import os
from PySide import QtCore, QtGui
from PySide.QtCore import QObject, Signal, Slot
import unicodedata
import yaml


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
    #@TODO make a slot in "Model settings" tab's dialog
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
        self.get_model_yaml_properties(format_filename)
        self.variables.export_data_path = format_filename
        self.ui.text_model_data_folder.setText(format_filename)



    def get_model_yaml_properties(self, directory):
        """
        extracts the properties from the selected model folder
        :param directory: directory to search for model yaml file
        """
        for file in os.listdir(directory):
            if file.endswith(".yaml"):
                yaml_file = file
                break

        with open(os.path.join(directory, yaml_file), 'r') as stream:
            try:
                ml_classes = yaml.load(stream)# print(yaml.load(stream))
            except yaml.YAMLError as exc:
                print(exc)

        self.variables.classes = ml_classes







