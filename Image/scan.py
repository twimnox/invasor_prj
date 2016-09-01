# from __future__ import division
# from zope.interface.tests.test_interface import I
import xml.etree.cElementTree as xmlET
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PySide import QtCore, QtGui
from PySide.QtCore import QObject, Signal, Slot
from Utils.variables import Variables
from Model.classifier import Classifier




class Scan(QObject):

    # Write once and then Read-only global vars
    IMG_ROOT = ""
    EXPORT_DIR = ""
    PATCH_SIZE = 0
    PATCH_OVERLAP = 0

    def __init__(self, variables):
        super(Scan, self).__init__()
        self.variables = variables
        self.classifier = Classifier(variables)
        self.color_list = (0, 102, 51), (255, 255, 0), (255, 255, 204), (51, 255, 51), (51, 25, 0), (0, 51, 0), (0, 51, 25), (128, 128, 128), (255, 0, 255)

        # - Vegetation #0
        # - acacia #1
        # - dirt #2
        # - ShortHerbs #3
        # - Wood #4
        # - Pinhal
        # - Sobral
        # - RoadWay
        # - other_yellow

        global IMG_ROOT, PATCH_SIZE, PATCH_OVERLAP, EXPORT_DIR
        PATCH_SIZE = 200         # @TODO PATCH_SIZE =... on model loading

        # @TODO PATCH_OVERLAP =... %percentage on model loading

    #SIGNALS
    update_layers_list = Signal()



        #SLOTS

    def scan_img(self):
        """
        Main function of this class
        creates XML templates for the image scanning outputs
        """
        xml_root, xml_classes_fields = self.create_xml_template()
        self.scanner(xml_root, xml_classes_fields)
        self.update_layers_list.emit()



    def create_xml_template(self):
        """
        creates XML tree for the image patches classification
        :return: XML root and classes XML elements
        """
        root = xmlET.Element("root")

        params = xmlET.SubElement(root, "parameters")
        xmlET.SubElement(params, "patch_size").text = self.PATCH_SIZE
        xmlET.SubElement(params, "patch_overlap").text = self.PATCH_OVERLAP
        xmlET.SubElement(params, "source_image").text = self.variables.import_data_path


        cls_list = []
        #create a field for each class's roi_coordinates
        for cls in self.variables.classes:
            curr_class = xmlET.SubElement(root, "class" ,name = cls)
            cls_list.append(curr_class)

        # @TODO load the model path for further classification
        return (root, cls_list)



    def scanner(self, xml_root, cls_list):
        """
        Scans input image with a Width*Height window.
        The window is then classified in the selected Machine Learned model
        Produces a XML file with coordinates for each classified type
        """
        global IMG_ROOT

        img = cv2.imread(self.variables.import_data_path)
        height, width = img.shape[:2]
        x_p_ = y_p_ = 0
        x_patches = width//PATCH_SIZE
        y_patches = height//PATCH_SIZE
        total_cycle = x_patches*y_patches
        cycle = 0 # @TODO to be used in a progressbar

        classes_rect_cnt = np.uint8([0 for x in range(self.variables.NUMBER_OF_CLASSES)])
        # cls = Classifier(self.variables)

        image_placeholder = np.zeros((y_patches, x_patches, 3), np.uint8) #Generate image placeholder for classifications

        # CROPPING:
        # @TODO improve to process image borders and with patch overlap
        for x_p in range(0,x_patches):
            for y_p in range(0,y_patches):
                x_p_ = x_p * PATCH_SIZE
                y_p_ = y_p * PATCH_SIZE
                if y_p_+PATCH_SIZE < height and x_p_+PATCH_SIZE < width:
                    crop_img = img[y_p_:y_p_+PATCH_SIZE, x_p_:x_p_+PATCH_SIZE]
                    resize_img = cv2.resize(crop_img, (90, 90)) #@TODO remove this. the dimentions size must come automatically from PATCH_SIZE and must consider the crop size
                    predicted_class_ID = self.classifier.classify(resize_img)
                    # populate image_placeholder with colors representing each class
                    print x_p, "/", x_patches, y_p, "/", y_patches
                    image_placeholder[y_p, x_p, 2] = self.color_list[predicted_class_ID][0]
                    image_placeholder[y_p, x_p, 1] = self.color_list[predicted_class_ID][1]
                    image_placeholder[y_p, x_p, 0] = self.color_list[predicted_class_ID][2]


                    # coordXY = xmlET.SubElement(cls_list[predicted_class_ID], "rect", name = str(classes_rect_cnt[predicted_class_ID]))
                    # xmlET.SubElement(coordXY, "X").text = str(x_p_)
                    # xmlET.SubElement(coordXY, "Y").text = str(y_p_)

                    # classes_rect_cnt[predicted_class_ID] += 1
                    cycle += 1
                    print "cycle", cycle, "of total:", total_cycle

        #Write XML template to file:
        tree = xmlET.ElementTree(xml_root)
        if not (self.variables.export_data_path == "empty"):
            # tree.write(os.path.join(self.variables.export_data_path, "classification_output.xml"))
            cv2.imwrite(os.path.join(self.variables.export_data_path, "image_placeholder.jpg"), image_placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            resize_placeholder = cv2.resize()
            cv2.imshow("result", image_placeholder)
            #Now lets erode the image:
            kernel = np.ones((3,3), np.uint8)
            erosion = cv2.erode(image_placeholder, kernel, iterations = 1)
            cv2.imwrite(os.path.join(self.variables.export_data_path, "image_placeholder_eroded.jpg"), erosion, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            dilation = cv2.dilate(erosion, kernel, iterations = 1)
            cv2.imwrite(os.path.join(self.variables.export_data_path, "image_placeholder_dilated.jpg"), dilation, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


            #Now lets dilate the image:
        else:
            print "No output folder is defined!"
            # @TODO show dialog warning

















