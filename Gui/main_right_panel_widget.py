# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_right_panel_widget.ui'
#
# Created: Fri Apr  1 14:10:59 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_main_right_panel_widget(object):
    def setupUi(self, main_right_panel_widget):
        main_right_panel_widget.setObjectName("main_right_panel_widget")
        main_right_panel_widget.resize(817, 529)
        self.horizontalLayout = QtGui.QHBoxLayout(main_right_panel_widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_zoom_out = QtGui.QPushButton(main_right_panel_widget)
        self.btn_zoom_out.setMaximumSize(QtCore.QSize(30, 30))
        self.btn_zoom_out.setObjectName("btn_zoom_out")
        self.horizontalLayout.addWidget(self.btn_zoom_out)
        self.btn_zoom_in = QtGui.QPushButton(main_right_panel_widget)
        self.btn_zoom_in.setMinimumSize(QtCore.QSize(0, 0))
        self.btn_zoom_in.setMaximumSize(QtCore.QSize(30, 30))
        self.btn_zoom_in.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_zoom_in.setObjectName("btn_zoom_in")
        self.horizontalLayout.addWidget(self.btn_zoom_in)
        self.graphicsView = QtGui.QGraphicsView(main_right_panel_widget)
        self.graphicsView.setObjectName("graphicsView")
        self.horizontalLayout.addWidget(self.graphicsView)

        self.retranslateUi(main_right_panel_widget)
        QtCore.QMetaObject.connectSlotsByName(main_right_panel_widget)

    def retranslateUi(self, main_right_panel_widget):
        main_right_panel_widget.setWindowTitle(QtGui.QApplication.translate("main_right_panel_widget", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_zoom_out.setText(QtGui.QApplication.translate("main_right_panel_widget", "-", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_zoom_in.setText(QtGui.QApplication.translate("main_right_panel_widget", "+", None, QtGui.QApplication.UnicodeUTF8))

