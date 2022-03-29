import cv2
import numpy as np
from skimage.segmentation import random_walker
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import random

def get_marker(img, r1 = 0.55, r2 = 0.9):
    markers = np.zeros(img.shape, dtype = np.uint)
    m, n = img.shape
    rad1 = int(r1 * min(m, n)/2)
    rad2 = int(r2 * min(m, n)/2)
    x, y = int(m/2), int(n/2)
    for i in range(m):
        for j in range(n):
            if random.randint(0,100)%2 == 0 and (i-x)**2 + (j-y)**2 < rad1**2 and img[i][j] < 0:
                markers[i][j] = 1
            if random.randint(0,100)%2 == 0 and img[i][j] > 0:
                markers[i][j] = 2
            if random.randint(0,100)%2 == 0 and (i-x)**2 + (j-y)**2 > rad2**2 and img[i][j] < 0:
                markers[i][j] = 3

    return markers


def random_walker_segmentation(img):
    im = img.astype(dtype = np.float32)
    im_sc = rescale_intensity(im, out_range=(-1, 1))
    markers = get_marker(im_sc)
    labels = random_walker(im_sc, markers, beta=100, mode='bf')

    out = np.zeros(img.shape, dtype = np.uint8)
    
    out[labels == 1] = 255    
    return out

if __name__ == '__main__':
    img = cv2.imread('temp_output/step1.jpg', 0)
    
    out = random_walker_segmentation(img)
    cv2.imshow('Output', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()