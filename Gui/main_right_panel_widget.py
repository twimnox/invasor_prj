# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_right_panel_widget.ui'
#
# Created: Thu Mar 24 16:45:37 2016
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
        self.graphicsView = QtGui.QGraphicsView(main_right_panel_widget)
        self.graphicsView.setObjectName("graphicsView")
        self.horizontalLayout.addWidget(self.graphicsView)

        self.retranslateUi(main_right_panel_widget)
        QtCore.QMetaObject.connectSlotsByName(main_right_panel_widget)

    def retranslateUi(self, main_right_panel_widget):
        main_right_panel_widget.setWindowTitle(QtGui.QApplication.translate("main_right_panel_widget", "Form", None, QtGui.QApplication.UnicodeUTF8))

