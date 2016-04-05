# from __future__ import division
# from zope.interface.tests.test_interface import I
import xml.etree.cElementTree as xmlET
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
from Utils.variables import Variables


class Scan(object):

    # Write once and then Read-only global vars
    IMG_ROOT = ""
    EXPORT_DIR = ""
    PATCH_SIZE = 0
    PATCH_OVERLAP = 0


    def __init__(self, variables):
        super(ImageWidget, self).__init__()
        self.ui = Ui_main_right_panel_widget()
        self.ui.setupUi(self)
        self.variables = variables

        global IMG_ROOT, PATCH_SIZE, PATCH_OVERLAP, EXPORT_DIR
        IMG_ROOT = self.variables.import_data_path
        EXPORT_DIR = self.variables.export_data_path
        # @TODO PATCH_SIZE1 =... on model loading
        # @TODO PATCH_OVERLAP =... %percentage on model loading

        #SLOTS



    def create_xml(self):
        rect_cnt = np.uint8([0 for x in range(self.variables.NUMBER_OF_CLASSES)])
        root = xmlET.Element("roi_coordinates")
        xmlET.SubElement(root, "rect_", rect_cnt)
        # @TODO load number of classes and classes names from the yaml (must create the GUI interaction for this...)
        # @TODO load the model path for further classification
        return


    def scanner(self):
        """
        Scans input image with a Width*Height window.
        The window is then classified in the selected Machine Learned model
        """
        img = cv.imread(IMG_ROOT)
        height, width = img.shape[:2]
        x_p_ = y_p_ = 0
        x_patches = width//PATCH_SIZE
        y_patches = height//PATCH_SIZE
        total_cycle = x_patches*y_patches


        #XML vars


        # CROPPING:
        # @TODO improve with image borders
        for x_p in range(0,x_patches):
            for y_p in range(4,y_patches-4):
                x_p_ = x_p * PATCH_SIZE
                y_p_ = y_p * PATCH_SIZE
                if y_p_+PATCH_SIZE < ortho_height and x_p_+PATCH_SIZE < ortho_width:
                    crop_img = img[y_p_:y_p_+PATCH_SIZE, x_p_:x_p_+PATCH_SIZE]

                    # @TODO classify using model

                    cycle += 1

















