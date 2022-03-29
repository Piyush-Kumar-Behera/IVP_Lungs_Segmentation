import cv2
import numpy as np

def pass_filter(img, r = 7):
    m, n = img.shape

    k = 2*r + 1
    new_img = np.zeros((m+r*2, n+r*2), dtype = np.float32)
    new_img[r:r+m, r:r+n] = img
    output_img = np.zeros((m,n), dtype = np.float32)

    for i in range(0, m):
        for j in range(0, n):
            sub_img = new_img[i:i+k, j:j+k]
            output_img[i][j] = np.mean(sub_img)
    
    return output_img

def guided_filter(image, p_img, radius = 7, eps = 0.008):
    im = image.astype(dtype=np.float32)
    p = p_img.astype(dtype=np.float32)
    
    im = im/255
    p = p/255

    mI  = pass_filter(img = im, r=radius)

    mp  = pass_filter(img = p, r=radius)
    cI  = pass_filter(img = im*im, r=radius)
    cp = pass_filter(img = im*p, r=radius)

    vI   = cI - mI * mI
    covIp  = cp - mI * mp

    a      = covIp / (vI + eps)
    b      = mp - a * mI

    meana  = pass_filter(img=a, r=radius)
    meanb  = pass_filter(img=b, r=radius)

    q = meana * im + meanb
    q = q * 255

    final_out = q.astype(dtype = np.uint8)

    return final_out

if __name__ == '__main__':
    img = cv2.imread('images/test_image_1.jpeg', 0)
    
    out = guided_filter(img, img)
    cv2.imshow('Imput Image', img)
    cv2.imshow('Output Image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()