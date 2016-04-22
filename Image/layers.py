import xml.etree.cElementTree as xmlET
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from Utils.variables import Variables
from PySide import QtCore




class Layers(object):

    # Write once and then Read-only global vars
    IMG_ROOT = ""
    EXPORT_DIR = ""
    PATCH_SIZE = 0
    PATCH_OVERLAP = 0



    def __init__(self, variables):
        super(Layers, self).__init__()
        self.variables = variables

        global IMG_ROOT, PATCH_SIZE, PATCH_OVERLAP, EXPORT_DIR
        IMG_ROOT = self.variables.import_data_path
        EXPORT_DIR = self.variables.export_data_path
        self.color_list = (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 230, 0), (255, 255, 255), (0, 0, 0), (128, 128, 128)
        PATCH_SIZE = 200         # @TODO PATCH_SIZE =... on model loading

        # @TODO PATCH_OVERLAP =... %percentage on model loading

        #SLOTS

    @QtCore.Slot()
    def generate_all_layers(self):
        print "generating all layers..."
        tree = xmlET.ElementTree(file = os.path.join(self.variables.export_data_path, "classification_output.xml"))
        root = tree.getroot()
        img = cv2.imread(self.variables.import_data_path)

        for i in range(len(self.variables.classes)):
            query = "class[@name="+"\""+self.variables.classes[i]+"\""+"]/rect"

            for cls in root.findall(query):
                # print "Rect id", cls.get("name")
                x = int(cls[0].text) #X
                y = int(cls[1].text) #Y
                cv2.rectangle(img, (x, y), (x+200, y+200), self.color_list[i], 2) #@TODO 200 size must be dynamic
                cv2.imshow('img', img)

        # finaly display the layers:
        cv2.imwrite(os.path.join(self.variables.TMP_DIR, "layered_img.jpg"), img)





    def generate_selected_layers(self):
        checked_list = []

        for i in range(len(self.variables.layers_checkboxes)):
            if self.variables.layers_checkboxes.isChecked():
                self.generate_one_layer(self.variables.classes[i])
                # checked_list.append(self.variables.classes[i]) #classes[i] are checked. they are displayed in the same order as they are placed in the "classes" array from variables


    def generate_one_layer(self, class_name):
        tree = xmlET.ElementTree(file = os.path.join(self.variables.export_data_path, "classification_output.xml"))
        root = tree.getroot()

        for i in range(len(self.variables.classes)):
            query = "class[@name="+"\""+class_name+"\""+"]/rect"

            for cls in root.findall(query):
                print "Rect id", cls.get("name")
                print "X", cls[0].text #X
                print "Y", cls[1].text #Y

        # @TODO print squares over imported image


