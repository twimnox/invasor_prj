# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog_model.ui'
#
# Created: Wed Apr  6 16:06:14 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(600, 500)
        Dialog.setMinimumSize(QtCore.QSize(600, 500))
        Dialog.setMaximumSize(QtCore.QSize(600, 500))
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(40, 110, 141, 17))
        self.label_3.setObjectName("label_3")
        self.text_model_data_folder = QtGui.QLineEdit(Dialog)
        self.text_model_data_folder.setGeometry(QtCore.QRect(40, 130, 401, 27))
        self.text_model_data_folder.setObjectName("text_model_data_folder")
        self.btn_model_data_folder = QtGui.QPushButton(Dialog)
        self.btn_model_data_folder.setGeometry(QtCore.QRect(460, 130, 99, 27))
        self.btn_model_data_folder.setObjectName("btn_model_data_folder")
        self.btn_ok = QtGui.QPushButton(Dialog)
        self.btn_ok.setGeometry(QtCore.QRect(460, 430, 99, 27))
        self.btn_ok.setObjectName("btn_ok")
        self.btn_cancel = QtGui.QPushButton(Dialog)
        self.btn_cancel.setGeometry(QtCore.QRect(350, 430, 99, 27))
        self.btn_cancel.setObjectName("btn_cancel")
        self.listView_model_properties = QtGui.QListView(Dialog)
        self.listView_model_properties.setGeometry(QtCore.QRect(40, 200, 241, 251))
        self.listView_model_properties.setObjectName("listView_model_properties")
        self.label_4 = QtGui.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(40, 180, 141, 17))
        self.label_4.setObjectName("label_4")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Import Data", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Dialog", "Model Data Folder:", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_model_data_folder.setText(QtGui.QApplication.translate("Dialog", "Select Folder", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_ok.setText(QtGui.QApplication.translate("Dialog", "Ok", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_cancel.setText(QtGui.QApplication.translate("Dialog", "Cancel", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Dialog", "Model Properties:", None, QtGui.QApplication.UnicodeUTF8))

