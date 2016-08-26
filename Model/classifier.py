import cifar10_eval_test as cifar10_classify
import cv2
import numpy as np


class Classifier(object):

    def __init__(self, variables):
        super(Classifier, self).__init__()
        self.variables = variables

    def classify(self, patch):
        print self.variables.MODEL_IMAGE_SIZE
        cifar10_classify.init_tf(self.variables.MODEL_IMAGE_SIZE,
                                 # self.variables.NUMBER_OF_CLASSES - 1 , # if I have n classes, model must go from 0 to n-1 classes
                                 self.variables.NUMBER_OF_CLASSES,
                                 self.variables.model_folder_path)
        bin = self.convert_img_to_binary(patch)
        result = cifar10_classify.evaluate_one(binary_image=bin)
        return result

    def convert_img_to_binary(self, img):
        """
        converts an image to binary format: [red values][green values][blue values], in a top-down, left-right order.
        :param img: image to convert to binary
        """
        # @TODO make width and height dimensions dynamic with the program model
        # @TODO: further processing on cifar10.evaluate_one can be optimized here. for instance the [arr] shape and dtype

        width = self.variables.MODEL_IMAGE_SIZE
        height = self.variables.MODEL_IMAGE_SIZE
        # resized_image = cv2.resize(img, (width, height)) #unnecessary
        arr = np.uint8([0 for x in range(width*height*3)])
        one_color_bytes = width * height

        arr_cnt = 0
        for y in range(0, width):
            for x in range(0, height):
                arr[arr_cnt] = np.uint8(img[x, y, 2])  # R
                arr[arr_cnt + one_color_bytes] = np.uint8(img[x, y, 1])  # G
                arr[arr_cnt + 2*one_color_bytes] = np.uint8(img[x, y, 0])  # B

                arr_cnt += 1

        return arr


if __name__ == '__main__':
    import sys

    #Test code (for just 1 image):
    cls = Classifier("empty")
    # path = "/home/prtricardo/tensorflow_tmp/200x200_models/acacia10_test/just 1 image/acacia106.jpg"
    path = "/home/prtricardo/tese_ws/open_cv/calib_images_treshold/varios tamanhos imagens individuais/13-3-2016-1o/imagens_originais/Wood0/wood85.jpg"
    img = cv2.imread(path)
    img = cv2.resize(img, (180, 180))
    res = cls.classify(img)
    print "it's a", res, "!"

    # sys.exit(app.exec_())
