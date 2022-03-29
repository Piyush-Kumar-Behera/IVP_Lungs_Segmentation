import cv2
import numpy as np

def floodFill(img):

    im_in = 255 - img
    th,im_th = cv2.threshold(im_in,127,255,cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv

    return im_out

if __name__ == '__main__':
    img = cv2.imread('temp_output/step2.jpg', 0)
    out = floodFill(img)

    cv2.imshow('Input', img)
    cv2.imshow('Output', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()