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

def get_structuring_element():
    se = [[255,255,255,255,255,255,255,255,255],
        [255,255,255,255,255,255,255,255,255],
        [255,255,255,255,255,255,255,255,255],
        [255,255,255,255,255,255,255,255,255],
        [255,255,255,255,255,255,255,255,255],
        [255,255,255,255,255,255,255,255,255],
        [255,255,255,255,255,255,255,255,255],
        [255,255,255,255,255,255,255,255,255],
        [255,255,255,255,255,255,255,255,255]]

    se = np.ones((5,5)) * 255

    return se

def contour_smoothing(img):
    se = get_structuring_element()
    eroded = erode_this(img, se)
    dilated = dilate_this(eroded, se)
    out = dilated.astype(dtype = np.uint8)
    return out

if __name__ == '__main__':
    img = cv2.imread('temp_output/step3.jpg',0)
    out = contour_smoothing(img)

    cv2.imshow('Input Image', img)
    cv2.imshow('Output Image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








