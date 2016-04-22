# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_left_panel_widget.ui'
#
# Created: Fri Apr 22 15:38:16 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_main_left_panel_widget(object):
    def setupUi(self, main_left_panel_widget):
        main_left_panel_widget.setObjectName("main_left_panel_widget")
        main_left_panel_widget.resize(300, 529)
        self.formLayout = QtGui.QFormLayout(main_left_panel_widget)
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName("formLayout")
        self.label = QtGui.QLabel(main_left_panel_widget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label)
        self.layersScrollArea = QtGui.QScrollArea(main_left_panel_widget)
        self.layersScrollArea.setMinimumSize(QtCore.QSize(0, 120))
        self.layersScrollArea.setWidgetResizable(True)
        self.layersScrollArea.setObjectName("layersScrollArea")
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 280, 118))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.formLayout_2 = QtGui.QFormLayout(self.scrollAreaWidgetContents)
        self.formLayout_2.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_nothing_to_show = QtGui.QLabel(self.scrollAreaWidgetContents)
        font = QtGui.QFont()
        font.setFamily("Monospace")
        font.setPointSize(9)
        self.label_nothing_to_show.setFont(font)
        self.label_nothing_to_show.setObjectName("label_nothing_to_show")
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_nothing_to_show)
        self.layersScrollArea.setWidget(self.scrollAreaWidgetContents)
        self.formLayout.setWidget(1, QtGui.QFormLayout.SpanningRole, self.layersScrollArea)

        self.retranslateUi(main_left_panel_widget)
        QtCore.QMetaObject.connectSlotsByName(main_left_panel_widget)

    def retranslateUi(self, main_left_panel_widget):
        main_left_panel_widget.setWindowTitle(QtGui.QApplication.translate("main_left_panel_widget", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("main_left_panel_widget", "Show Classes:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_nothing_to_show.setText(QtGui.QApplication.translate("main_left_panel_widget", "No inputs loaded", None, QtGui.QApplication.UnicodeUTF8))

