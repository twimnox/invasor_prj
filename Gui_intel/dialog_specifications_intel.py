import Gui.dialog_specifications as ds
import os
from PySide import QtCore, QtGui
from PySide.QtCore import QObject, Signal, Slot
import unicodedata
import yaml


class Ui_Specifications_Interaction(QObject):

    def __init__(self, variables):
        QObject.__init__(self)

        self.variables = variables
        self.DataDialog = QtGui.QDialog()
        self.ui = ds.Ui_Dialog()
        self.ui.setupUi(self.DataDialog)

        #SLOTS
        self.ui.btn_ok.clicked.connect(self.ok_click)



    def ok_click(self):
        # self.update_model_data.emit()
        self.DataDialog.close()

    def open_dialog(self):
        """
        Opens the Load Data dialog
        """
        self.DataDialog.show()

    @QtCore.Slot()
    def update_specs(self):
        self.display_preview_model_classes()
        self.ui.text_model_precision.setText("-999")
        self.ui.text_model_data_folder.setText(self.variables.model_folder_path)


    def display_preview_model_classes(self):
        """
        displays the selected model classes in the listview from specifications dialog
        """
        ml_classes = self.variables.classes
        model = QtGui.QStandardItemModel(self.ui.listView_model_classes)
        for cls in ml_classes:
            item = QtGui.QStandardItem(cls)
            # item.setCheckable(True)
            model.appendRow(item)
        self.ui.listView_model_classes.setModel(model)












