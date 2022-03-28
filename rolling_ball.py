# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon Mar 28 14:37:52 2022

# @author: dhyeybm
# """



import cv2;
import numpy as np;

# Read image



def rolling_ball(im_in):
    

    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
    
    # th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);
    th,im_th = cv2.threshold(im_in,127,255,cv2.THRESH_BINARY_INV)

    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    cv2.imshow("Input Image", im_in)
    cv2.imshow("Thresholded Image", im_th)
    cv2.imshow("Floodfilled Image", im_floodfill)
    cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    cv2.imshow("Foreground", im_out)
    cv2.waitKey(0)
    return im_out

# Display images.
# im_in = cv2.imread("./images/test_rolling.png", cv2.IMREAD_GRAYSCALE);
# im_out = rolling_ball(im_in)

