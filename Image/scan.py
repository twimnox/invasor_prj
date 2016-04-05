#from __future__ import division
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os


class Scan(object):

    def __init__(self, variables):
        super(ImageWidget, self).__init__()
        self.ui = Ui_main_right_panel_widget()
        self.ui.setupUi(self)
        self.variables = variables

        #SLOTS
        self.ui.btn_zoom_in.clicked.connect(self.zoom_in)
        self.ui.btn_zoom_out.clicked.connect(self.zoom_out)


    acacia_fld = 'Acacia'
    ground_fld = 'Ground'
    vegetation_fld = 'Vegetation'

    #create folders:
    # newpath = acacia_fld
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)
    # newpath = ground_fld
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)
    # newpath = vegetation_fld
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)

    PATCH_SIZE = 150
    FOLDER_SIZE = 1000
    NBLACK = 10

    img = cv.imread("DJI_0630.JPG")
    ortho_height, ortho_width = img.shape[:2]
    acacia_cnt = vegetation_cnt = ground_cnt = 0
    invalids = 0
    cycle = 0
    x_p_ = y_p_ = 0
    x_patches = ortho_width//PATCH_SIZE
    y_patches = ortho_height//PATCH_SIZE
    total_cycle = x_patches*y_patches

    print 'width:', ortho_width, " height:", ortho_height


    def patch_hist(patch):
        width, height = patch.shape[:2]
        sum_r = sum_b = sum_g = 0
        type = 0
        black_pixels = 0

        for x in range(0,width):
            for y in range(0,height):
                if patch[x,y,0] == 0 and patch[x,y,1] == 0 and patch[x,y,2] == 0:
                    black_pixels += 1
                else:
                    sum_r = sum_r + patch[x,y,2]
                    sum_g = sum_g + patch[x,y,1]
                    sum_b = sum_b + patch[x,y,0]

        if black_pixels >= NBLACK:
            type = 99  # has black regions. Doesnt matter
        elif sum_b != 0:
            calc = (((sum_r+sum_g)/2)/float(sum_b))
            calc2 = sum_r+sum_g+sum_b

            # ground
            if calc < 1.099 and calc2 > 1200000:
                type = 0

            # acacia
            elif calc >= 1.80:
                type = 2

            # other vegetation
            elif calc >= 1.15 and calc <= 1.39 and calc2 < 680000:
                type = 1
            else:
                type = 99
        return type

    for x_p in range(0,x_patches):
        for y_p in range(4,y_patches-4):
            x_p_ = x_p * PATCH_SIZE
            y_p_ = y_p * PATCH_SIZE
            if y_p_+PATCH_SIZE < ortho_height and x_p_+PATCH_SIZE < ortho_width:
                crop_img = img[y_p_:y_p_+PATCH_SIZE, x_p_:x_p_+PATCH_SIZE]
                type = patch_hist(crop_img)

                if type == 0: #GROUND
                    if not os.path.exists('Ground'+str(ground_cnt//FOLDER_SIZE)):
                        ground_fld = 'Ground'+str(ground_cnt//FOLDER_SIZE)
                        os.makedirs(ground_fld)
                    #cv.imwrite(os.path.join(acacia_fld, 'acacia'+str(acacia_cnt)+'.jpg'), crop_img)
                    cv.imwrite(os.path.join(ground_fld, 'ground'+str(ground_cnt)+'.jpg'), crop_img, [cv.IMWRITE_JPEG_QUALITY, 95])
                    ground_cnt += 1
                elif type == 1: #VEGETATION
                    if not os.path.exists('Vegetation'+str(vegetation_cnt//FOLDER_SIZE)):
                        vegetation_fld = 'Vegetation'+str(vegetation_cnt//FOLDER_SIZE)
                        os.makedirs(vegetation_fld)
                    #cv.imwrite(os.path.join(vegetation_fld, 'vegetation'+str(vegetation_cnt)+'.jpg'), crop_img)
                    cv.imwrite(os.path.join(vegetation_fld, 'vegetation'+str(vegetation_cnt)+'.jpg'), crop_img, [cv.IMWRITE_JPEG_QUALITY, 95])
                    vegetation_cnt += 1
                elif type == 2: #ACACIA
                    if not os.path.exists('Acacia'+str(acacia_cnt//FOLDER_SIZE)):
                        acacia_fld = 'Acacia'+str(acacia_cnt//FOLDER_SIZE)
                        os.makedirs(acacia_fld)
                    #cv.imwrite(os.path.join(acacia_fld, 'acacia'+str(acacia_cnt)+'.jpg'), crop_img)
                    cv.imwrite(os.path.join(acacia_fld, 'acacia'+str(acacia_cnt)+'.jpg'), crop_img, [cv.IMWRITE_JPEG_QUALITY, 95])
                    acacia_cnt += 1
                elif type == 99:
                    print "found invalid patch at", cycle
                    invalids += 1
                print 'cycle', cycle, 'of total cycles', total_cycle
                cycle += 1

    print 'Ground:', ground_cnt, 'Vegetation:', vegetation_cnt, 'Acacias:', acacia_cnt, 'Invalids:', invalids
