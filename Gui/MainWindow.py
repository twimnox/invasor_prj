# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created: Mon Mar 21 11:03:58 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtGui.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(180, 10, 611, 531))
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.imglabel = QtGui.QLabel(self.frame)
        self.imglabel.setGeometry(QtCore.QRect(10, 10, 590, 510))
        self.imglabel.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignTop)
        self.imglabel.setObjectName("imglabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        self.menuModel = QtGui.QMenu(self.menubar)
        self.menuModel.setObjectName("menuModel")
        self.menuImport = QtGui.QMenu(self.menubar)
        self.menuImport.setObjectName("menuImport")
        self.menuAbout = QtGui.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionImport = QtGui.QAction(MainWindow)
        self.actionImport.setObjectName("actionImport")
        self.actionData_Import = QtGui.QAction(MainWindow)
        self.actionData_Import.setEnabled(True)
        self.actionData_Import.setObjectName("actionData_Import")
        self.actionMaps = QtGui.QAction(MainWindow)
        self.actionMaps.setObjectName("actionMaps")
        self.actionModel = QtGui.QAction(MainWindow)
        self.actionModel.setObjectName("actionModel")
        self.actionTest = QtGui.QAction(MainWindow)
        self.actionTest.setObjectName("actionTest")
        self.actionAuthor = QtGui.QAction(MainWindow)
        self.actionAuthor.setObjectName("actionAuthor")
        self.actionVersion = QtGui.QAction(MainWindow)
        self.actionVersion.setObjectName("actionVersion")
        self.actionImport_Model = QtGui.QAction(MainWindow)
        self.actionImport_Model.setObjectName("actionImport_Model")
        self.menuModel.addAction(self.actionImport_Model)
        self.menuModel.addAction(self.actionImport)
        self.menuModel.addAction(self.actionTest)
        self.menuModel.addSeparator()
        self.menuImport.addAction(self.actionData_Import)
        self.menuAbout.addAction(self.actionAuthor)
        self.menuAbout.addAction(self.actionVersion)
        self.menubar.addAction(self.menuImport.menuAction())
        self.menubar.addAction(self.menuModel.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.imglabel.setText(QtGui.QApplication.translate("MainWindow", "No Image Loaded", None, QtGui.QApplication.UnicodeUTF8))
        self.menuModel.setTitle(QtGui.QApplication.translate("MainWindow", "Model", None, QtGui.QApplication.UnicodeUTF8))
        self.menuImport.setTitle(QtGui.QApplication.translate("MainWindow", "Data", None, QtGui.QApplication.UnicodeUTF8))
        self.menuAbout.setTitle(QtGui.QApplication.translate("MainWindow", "About", None, QtGui.QApplication.UnicodeUTF8))
        self.actionImport.setText(QtGui.QApplication.translate("MainWindow", "Specifications", None, QtGui.QApplication.UnicodeUTF8))
        self.actionData_Import.setText(QtGui.QApplication.translate("MainWindow", "Import...", None, QtGui.QApplication.UnicodeUTF8))
        self.actionMaps.setText(QtGui.QApplication.translate("MainWindow", "Maps", None, QtGui.QApplication.UnicodeUTF8))
        self.actionModel.setText(QtGui.QApplication.translate("MainWindow", "Model", None, QtGui.QApplication.UnicodeUTF8))
        self.actionTest.setText(QtGui.QApplication.translate("MainWindow", "Test", None, QtGui.QApplication.UnicodeUTF8))
        self.actionAuthor.setText(QtGui.QApplication.translate("MainWindow", "Author", None, QtGui.QApplication.UnicodeUTF8))
        self.actionVersion.setText(QtGui.QApplication.translate("MainWindow", "Version", None, QtGui.QApplication.UnicodeUTF8))
        self.actionImport_Model.setText(QtGui.QApplication.translate("MainWindow", "Import Model", None, QtGui.QApplication.UnicodeUTF8))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

