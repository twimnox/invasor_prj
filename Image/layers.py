import xml.etree.cElementTree as xmlET
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from Utils.variables import Variables




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
        PATCH_SIZE = 200         # @TODO PATCH_SIZE =... on model loading

        # @TODO PATCH_OVERLAP =... %percentage on model loading

        #SLOTS



