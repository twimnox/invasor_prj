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
        self.color_list = (0, 51, 25), (255, 255, 0), (255, 255, 204), (51, 255, 51), (51, 25, 0), (18, 18, 235), (235, 18, 18), (128, 128, 128), (255, 0, 255)
        #          vegetation.v escuro acacia-amarelo  dirt-creme     s.herbs-v.clar  wood-castanho  pinhal-blue  sobral-verm    roadway-cinza   other yellow - rosa choque
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
        xml_root_2_filtered = xml_root

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

        # Write XML template to file:
        # tree = xmlET.ElementTree(xml_root)
        if not (self.variables.export_data_path == "empty"):
            # tree.write(os.path.join(self.variables.export_data_path, "classification_output.xml"))
            self.map_pixeled_image_to_xml(xml_root, cls_list, image_placeholder, "classification_output.xml")
            cv2.imwrite(os.path.join(self.variables.export_data_path, "image_placeholder_.png"), image_placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # @TODO in the layer class, make a layered image with the filtered output:
            filtered_img = self.noise_filter(image_placeholder)
            self.map_pixeled_image_to_xml(xml_root_2_filtered, cls_list, filtered_img, "classification_output_filtered.xml")
            # Lets save our filtered image...
            cv2.imwrite(os.path.join(self.variables.export_data_path, "image_placeholder_filtered.png"), filtered_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # resize_placeholder = cv2.resize(image_placeholder, (100, 100))
            # cv2.imshow("result", resize_placeholder)
            #Now lets erode the image:
            # kernel = np.ones((3,3), np.uint8)
            # erosion = cv2.erode(image_placeholder, kernel, iterations = 1)
            # cv2.imwrite(os.path.join(self.variables.export_data_path, "image_placeholder_eroded.png"), erosion, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # dilation = cv2.dilate(erosion, kernel, iterations = 1)
            # cv2.imwrite(os.path.join(self.variables.export_data_path, "image_placeholder_dilated.png"), dilation, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


            #Now lets dilate the image:
        else:
            print "No output folder is defined!"
            # @TODO show dialog warning


    # uses the pixeled image to generate the XML
    def map_pixeled_image_to_xml(self, xml_root, cls_list, pixel_img, xml_filename):
        """
        Scans input image with a Width*Height window.
        The window is then classified in the selected Machine Learned model
        Produces a XML file with coordinates for each classified type
        """
        global IMG_ROOT

        height, width = pixel_img.shape[:2]
        x_patches = width
        y_patches = height
        total_cycle = x_patches*y_patches
        cycle = 0 # @TODO to be used in a progressbar

        classes_rect_cnt = np.uint8([0 for x in range(self.variables.NUMBER_OF_CLASSES)])



        for x_p in range(0, width):
            for y_p in range(0, height):
                x_p_ = x_p * PATCH_SIZE
                y_p_ = y_p * PATCH_SIZE

                predict_class_ID = self.get_class_from_color(pixel_img[y_p, x_p])

                coordXY = xmlET.SubElement(cls_list[predict_class_ID], "rect", name = str(classes_rect_cnt[predict_class_ID]))
                xmlET.SubElement(coordXY, "X").text = str(x_p_)
                xmlET.SubElement(coordXY, "Y").text = str(y_p_)

                classes_rect_cnt[predict_class_ID] += 1
                cycle += 1
                print "cycle", cycle, "of total:", total_cycle




        #Write XML template to file:
        tree = xmlET.ElementTree(xml_root)
        if not (self.variables.export_data_path == "empty"):
            tree.write(os.path.join(self.variables.export_data_path, xml_filename))

        else:
            print "No output folder is defined!"
            # @TODO show dialog warning


    def noise_filter(self, input_img):
        height, width = input_img.shape[:2]
        img_to_filter = input_img.copy()

        for x_p in range(0, width):
            for y_p in range(0, height):
                classes_presence_cnt = np.uint8([0 for x in range(self.variables.NUMBER_OF_CLASSES)])
                pre_classification = self.get_class_from_color(img_to_filter[y_p, x_p])
                # default searching range (does a 3*3 region):
                x_low = y_low = -1
                x_high = y_high = 2
                # check if we can search the edges:
                if x_p == 0:
                    x_low = 0
                if x_p == width - 1: # 0 already counts in the array
                    x_high = 1
                if y_p == 0:
                    y_low = 0
                if y_p == height - 1: # 0 already counts in the array
                    y_high = 1
                # count the presence of classes within a 3*3 region, where the center is the img_to_filter[x_p, y_p]
                for x_k in range(x_low, x_high):
                    for y_k in range(y_low, y_high):
                        # see which class we find while searching...
                        print y_p + y_k, x_p + x_k
                        curr_class = self.get_class_from_color(img_to_filter[y_p + y_k, x_p + x_k])
                        # count the occurence of each class:
                        print "counted a class"
                        classes_presence_cnt[curr_class] += 1

                # If we only had one instance of the pre classified class, we change it with the class with most counts withing the 3*3 region
                if classes_presence_cnt[pre_classification] == 1:
                    print "changed a pixel!"
                    max_value = max(classes_presence_cnt)
                    max_indexes = [i for i, j in enumerate(classes_presence_cnt) if j == max_value] # (can be more than 1 max values!)
                    max_index = max_indexes[0] # we get the 1st instance of the max values @TODO to be improved
                    print "max index is ", max_index
                    img_to_filter[y_p, x_p][0] = self.color_list[max_index][2]
                    img_to_filter[y_p, x_p][1] = self.color_list[max_index][1]
                    img_to_filter[y_p, x_p][2] = self.color_list[max_index][0]

        return img_to_filter


    # receives a pixel and returns its INTEGER classification
    def get_class_from_color(self, pixel):

        for i in range(0, self.variables.NUMBER_OF_CLASSES):
            if (self.color_list[i][2] == pixel[0] and self.color_list[i][1] == pixel[1] and self.color_list[i][0] == pixel[2]):
                return i

        "nao apanhou as classes todas..."
        return -1














