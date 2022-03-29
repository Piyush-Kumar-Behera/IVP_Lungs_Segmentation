import cv2
import numpy as np

def extract_region(img, mask):
    m, n = img.shape

    out = np.zeros(img.shape, dtype = np.uint8)
    for i in range(m):
        for j in range(n):
            if mask[i][j] == 255:
                out[i][j] = img[i][j]
    return out

if __name__ == '__main__':
    img = cv2.imread('images/test_image_1.jpeg', 0)
    mask = cv2.imread('temp_output/step4.jpg', 0)

    out = extract_region(img, mask)

    cv2.imshow('Output Image1', img)
    cv2.imshow('Output Image2', mask)
    cv2.imshow('Output Image5', out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()