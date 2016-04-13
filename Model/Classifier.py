import cifar10_eval_test as cifar10_classify
import cv2
import numpy as np


class Classifier(object):

    def __init__(self, variables):
        super(Classifier, self).__init__()
        self.variables = variables

    def classify(self, patch):
        bin = self.convert_img_to_binary(patch)
        result = cifar10_classify.evaluate_one(binary_image=bin)
        return result

    def convert_img_to_binary(self, img):
        """
        converts an image to binary format: [red values][green values][blue values], in a top-down, left-right order.
        :param img: image to convert to binary
        """
        # @TODO make width and height dimentions dynamic with the program model

        width = 200
        height = 200
        resized_image = cv2.resize(img, (width, height)) #unnecessary
        arr = np.uint8([0 for x in range(width*height*3)])
        one_color_bytes = width * height

        arr_cnt = 0
        for y in range(0, width):
            for x in range(0, height):
                arr[arr_cnt] = np.uint8(resized_image[x, y, 2])  # R
                arr[arr_cnt + one_color_bytes] = np.uint8(resized_image[x, y, 1])  # G
                arr[arr_cnt + 2*one_color_bytes] = np.uint8(resized_image[x, y, 0])  # B

                arr_cnt += 1

        return arr


if __name__ == '__main__':
    import sys

    #Test code:
    cls = Classifier("empty")
    path = "/home/prtricardo/tese_ws/open_cv/acacia_model/acacia_.jpg"
    img = cv2.imread(path)
    cls.classify(img)

    sys.exit(app.exec_())
