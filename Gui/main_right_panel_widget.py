# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_right_panel_widget.ui'
#
# Created: Thu Mar 24 13:12:09 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_main_right_panel_widget(object):
    def setupUi(self, main_right_panel_widget):
        main_right_panel_widget.setObjectName("main_right_panel_widget")
        main_right_panel_widget.resize(782, 552)
        self.horizontalLayout = QtGui.QHBoxLayout(main_right_panel_widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.img_scrollArea = QtGui.QScrollArea(main_right_panel_widget)
        self.img_scrollArea.setWidgetResizable(True)
        self.img_scrollArea.setObjectName("img_scrollArea")
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 762, 532))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.scrollAreaWidgetContents)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.imglabel = QtGui.QLabel(self.scrollAreaWidgetContents)
        self.imglabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imglabel.setObjectName("imglabel")
        self.horizontalLayout_2.addWidget(self.imglabel)
        self.img_scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout.addWidget(self.img_scrollArea)

        self.retranslateUi(main_right_panel_widget)
        QtCore.QMetaObject.connectSlotsByName(main_right_panel_widget)

    def retranslateUi(self, main_right_panel_widget):
        main_right_panel_widget.setWindowTitle(QtGui.QApplication.translate("main_right_panel_widget", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.imglabel.setText(QtGui.QApplication.translate("main_right_panel_widget", "No Image Loaded", None, QtGui.QApplication.UnicodeUTF8))

