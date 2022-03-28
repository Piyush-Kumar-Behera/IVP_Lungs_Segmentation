import cv2
import numpy as np

def generate_hist(img):
    m, n = img.shape
    hist = np.zeros(256)

    for i in range(m):
        for j in range(n):
            hist[img[i][j]] += 1
    
    return hist

def get_score(img, th, hist):
    w0 = np.sum(hist[:th])/np.sum(hist)
    w1 = np.sum(hist[th:])/np.sum(hist)

    v0 = np.var(hist[:th])
    v1 = np.var(hist[th:])

    return w0*v0 + w1*v1

def get_threshold(img, hist):
    scores = []
    for th in range(1,254):
        scores.append(get_score(img, th, hist))
    
    return np.argmin(scores) + 1

def otsu_thresholding(img):
    m, n = img.shape
    hist = generate_hist(img)
    thresh = get_threshold(img, hist)
    out = np.zeros(img.shape, dtype=np.uint8)

    for i in range(m):
        for j in range(n):
            if img[i][j] > thresh:
                out[i][j] = 255
    
    return out

if __name__ == '__main__':
    img = cv2.imread('images/test_image_1.jpeg', 0)
    
    out = otsu_thresholding(img)
    cv2.imshow('Imput Image', img)
    cv2.imshow('Output Image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()