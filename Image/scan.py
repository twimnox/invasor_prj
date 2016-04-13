# from __future__ import division
# from zope.interface.tests.test_interface import I
import xml.etree.cElementTree as xmlET
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from Utils.variables import Variables
from Model.classifier import Classifier



class Scan(object):

    # Write once and then Read-only global vars
    IMG_ROOT = ""
    EXPORT_DIR = ""
    PATCH_SIZE = 0
    PATCH_OVERLAP = 0


    def __init__(self, variables):
        super(Scan, self).__init__()
        self.variables = variables

        global IMG_ROOT, PATCH_SIZE, PATCH_OVERLAP, EXPORT_DIR
        IMG_ROOT = self.variables.import_data_path
        EXPORT_DIR = self.variables.export_data_path
        PATCH_SIZE = 200         # @TODO PATCH_SIZE =... on model loading

        # @TODO PATCH_OVERLAP =... %percentage on model loading

        #SLOTS

    def scan_img(self):
        """
        Main function of this class
        creates XML templates for the image scanning outputs
        """
        xml_root, xml_classes_fields = self.create_xml_template()
        self.scanner(xml_root, xml_classes_fields)



    def create_xml_template(self):
        """
        creates XML tree for the image patches classification
        :return: XML root and classes XML elements
        """
        root = xmlET.Element("root")

        params = xmlET.SubElement(root, "parameters")
        xmlET.SubElement(params, "patch_size").text = self.PATCH_SIZE

        cls_list = []
        #create a field for each class's roi_coordinates
        for cls in self.variables.classes:
            curr_class = xmlET.SubElement(root, cls)
            cls_list.append(curr_class)

        # @TODO load the model path for further classification
        return (root, cls_list)



    def scanner(self, xml_root, cls_list):
        """
        Scans input image with a Width*Height window.
        The window is then classified in the selected Machine Learned model
        Produces a XML file with coordinates for each classified type
        """
        img = cv2.imread(IMG_ROOT)
        height, width = img.shape[:2]
        x_p_ = y_p_ = 0
        x_patches = width//PATCH_SIZE
        y_patches = height//PATCH_SIZE
        total_cycle = x_patches*y_patches
        cycle = 0 # @TODO to be used in a progressbar

        classes_rect_cnt = np.uint8([0 for x in range(self.variables.NUMBER_OF_CLASSES)])
        cls = Classifier("empty")


        # CROPPING:
        # @TODO improve to process image borders and with patch overlap
        for x_p in range(0,x_patches):
            for y_p in range(0,y_patches):
                x_p_ = x_p * PATCH_SIZE
                y_p_ = y_p * PATCH_SIZE
                if y_p_+PATCH_SIZE < height and x_p_+PATCH_SIZE < width:
                    crop_img = img[y_p_:y_p_+PATCH_SIZE, x_p_:x_p_+PATCH_SIZE]
                    img = cv2.resize(crop_img, (180, 180)) #@TODO remove this. the dimentions size must come automatically from PATCH_SIZE and must consider the crop size
                    predicted_class_ID = cls.classify(img)
                    coordXY = xmlET.SubElement(cls_list[predicted_class_ID], ("_rect_", classes_rect_cnt[predicted_class_ID]))
                    xmlET.SubElement(coordXY, "X").text = x_p_
                    xmlET.SubElement(coordXY, "Y").text = y_p_

                    classes_rect_cnt[predicted_class_ID] += 1
                    cycle += 1

        #Write XML template to file:
        tree = xmlET.ElementTree(root)
        tree.write("predicted_rectangles.xml") #os.join(outputfolder, predicted_rectangles.xml)


















