import cv2
import numpy as np
import time
def density_calc(img, r=25):
    m, n = img.shape

    k = 2*r + 1
    new_img = np.zeros((m+r*2, n+r*2), dtype = np.float32)
    new_img[r:r+m, r:r+n] = img
    output_img = np.zeros((m,n), dtype = np.float32)

    for i in range(0, m):
        for j in range(0, n):
            sub_img = new_img[i:i+k, j:j+k]
            output_img[i][j] = np.mean(sub_img)/255
    
    return output_img

def region_growing_helper(Seed, dense, T):
    r,c = dense.shape
    res = np.zeros((dense.shape),dtype=np.uint8) 
    vis = np.zeros((dense.shape),dtype=np.uint8)


    dx = [-1,0,0,1]
    dy = [0,-1,1,0]
    
    for i in range(len(Seed)):
        queue = []
        seed = Seed[i]
        queue.append(seed)
        vis[seed] = 1
        res[seed] = 255
        while len(queue) > 0:
            x,y = queue.pop(0)
            for i in range(4):
                nx = x+dx[i]
                ny = y+dy[i]
                if nx<r and ny<c and nx>=0 and ny>=0 and vis[nx,ny] == 0 and dense[nx, ny] < T:
                    res[nx][ny] = 255
                    vis[nx][ny] = 1
                    queue.append((nx,ny))
    return res

def region_growing(img):
    seed = [(0,0),(0,img.shape[1]-1),(img.shape[0]-1,0)
        ,(img.shape[0]-1,img.shape[1]-1)]

    dense = density_calc(img, r=9)

    out = region_growing_helper(seed, dense, T = 0.3) ## Threshold should be >1.5*otsu_threshold

    res = (out == 0) * img
    return res

if __name__ == '__main__':
    img = cv2.imread('temp_output/step2.jpg', 0)

    out = region_growing(img)

    cv2.imshow('Imput Image', img)
    cv2.imshow('Output Image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()