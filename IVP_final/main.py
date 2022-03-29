from denoise_image import *
from otsu_binarization import *
from random_walker_segmentation import *
from floodFill import *
from contour_smoothing import *
from extract_region import *
from region_growing import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = cv2.imread('images/test_image_0.png', 0)
    
    out1 = guided_filter(img, img)
    out2 = otsu_thresholding(out1)
    out7 = region_growing(out2)
    out3 = random_walker_segmentation(out7)
    out4 = floodFill(out3)
    out5 = contour_smoothing(out4)
    out6 = extract_region(img, out5)

    cv2.imshow('Imput Image', img)
    cv2.imshow('Output Image1', out1)
    cv2.imshow('Output Image2', out2)
    cv2.imshow('Output Image7', out7)
    cv2.imshow('Output Image3', out3)
    cv2.imshow('Output Image4', out4)
    cv2.imshow('Output Image5', out5)
    cv2.imshow('Output Image6', out6)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('temp_output/step1.jpg', out1)
    cv2.imwrite('temp_output/step2.jpg', out2)
    cv2.imwrite('temp_output/step7.jpg', out7)
    cv2.imwrite('temp_output/step3.jpg', out3)
    cv2.imwrite('temp_output/step4.jpg', out4)
    cv2.imwrite('temp_output/step5.jpg', out5)
    cv2.imwrite('temp_output/step6.jpg', out6)
