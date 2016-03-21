import Gui.dialog_load_data as dld
from PySide import QtCore, QtGui


class Communicate(QtCore.QObject):
    # create a new signal on the fly and name it 'speak'
    speak = QtCore.Signal(int)

class Ui_Data_Dialog_interaction(object):


    def __init__(self, variables):
        self.variables = variables
        #print "ola", self.variables.export_data_path
        self.DataDialog = QtGui.QDialog()
        self.ui = dld.Ui_Dialog()
        self.ui.setupUi(self.DataDialog)

        #Signals&Slots
        # someone = Communicate(self.ui.btn_data_import_path.clicked)
        # someone.speak.connect(self.open_dialog())
        # someone.speak.emit(1)

        self.ui.btn_data_import_path.clicked.connect(self.open_file_dialog_import)
        self.ui.btn_data_output_folder.clicked.connect(self.open_file_dialog_export)


    def open_dialog(self):
        """
        Opens the Load Data dialog
        """

        self.DataDialog.show()

    def open_file_dialog_import(self):
        """
        Opens file browser prompt

        """
        fileDialog = QtGui.QFileDialog()
        fileDialog.setNameFilters("Image files (*.png *.tif, *.jpg)")
        fileDialog.setFileMode(QtGui.QFileDialog.AnyFile)
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
        filename = str(fileDialog.getOpenFileName()).encode('utf-8')


        self.variables.export_data_path = filename
        self.ui.text_data_output_folder.setText(filename)








