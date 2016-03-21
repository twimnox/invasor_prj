import Gui.dialog_load_data as dld
from PySide import QtCore, QtGui

class Ui_Data_Dialog_interaction(object):

    def open_dialog(self):
        self.DataDialog = QtGui.QDialog()
        ui = dld.Ui_Dialog()
        ui.setupUi(self.DataDialog)
        self.DataDialog.show()


