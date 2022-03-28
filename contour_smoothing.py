# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:57:40 2022

@author: sured
"""

import cv2
import numpy as np

def dilate_this(img, structuring_kernel):

    img = ((img >= 127)*255).astype('uint8')
    orig_shape = img.shape
    pad_width = len(structuring_kernel) - 2
    image_pad = np.pad(array=img, pad_width=pad_width, mode='constant')
    pimg_shape = image_pad.shape
    h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])
    
    flat_submatrices = np.array([
        image_pad[i:(i + len(structuring_kernel) ), j:(j + len(structuring_kernel) )]
        for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)
    ])

    image_dil = np.array([255 if (i == np.array(structuring_kernel)).any() else 0 for i in flat_submatrices])
    image_dil = image_dil.reshape(orig_shape)

    return image_dil

def erode_this(img, structuring_kernel):

    img = ((img >= 127)*255).astype('uint8')
    orig_shape = img.shape
    pad_width = len(structuring_kernel) - 2
    image_pad = np.pad(array=img, pad_width=pad_width, mode='constant')
    pimg_shape = image_pad.shape
    h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])

    flat_submatrices = np.array([
        image_pad[i:(i + len(structuring_kernel) ), j:(j + len(structuring_kernel) )]
        for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)
    ])

    image_erode = np.array([255 if (i == np.array(structuring_kernel)).all() else 0 for i in flat_submatrices])
    image_erode = image_erode.reshape(orig_shape)

    return image_erode

img_name = 'test_image_1'
img_extracted = cv2.imread(f'{img_name}_extracted.jpeg',0) ## image result of hole filling should be read

se = [[255,255,255,255,255,255,255,255,255],
      [255,255,255,255,255,255,255,255,255],
      [255,255,255,255,255,255,255,255,255],
      [255,255,255,255,255,255,255,255,255],
      [255,255,255,255,255,255,255,255,255],
      [255,255,255,255,255,255,255,255,255],
      [255,255,255,255,255,255,255,255,255],
      [255,255,255,255,255,255,255,255,255],
      [255,255,255,255,255,255,255,255,255]]

eroded = erode_this(img_extracted, se)
dilated = dilate_this(eroded, se)
cv2.imwrite(f'{img_name}_contour_smoothed.jpeg',dilated)








