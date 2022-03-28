# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:42:26 2022

@author: sured
"""

import cv2
import numpy as np


def region_growing(img, Seed, T):
    r,c = img.shape
    res = np.zeros((img.shape),dtype=np.uint8) 
    vis = np.zeros((img.shape),dtype=np.uint8)


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
                if nx<r and ny<c and nx>=0 and ny>=0 and vis[nx,ny] == 0 and (img[nx,ny].astype(int)<T):
                    res[nx][ny] = 255
                    vis[nx][ny] = 1
                    queue.append((nx,ny))
    return res

img_name = 'test_image_1'
img = cv2.imread(f'{img_name}.jpeg',0)
seed = [(0,0),(0,img.shape[1]-1),(img.shape[0]-1,0)
        ,(img.shape[0]-1,img.shape[1]-1)]

out = region_growing(img, seed, T = 220) ## Threshold should be >1.5*otsu_threshold
cv2.imwrite(f"{img_name}_background_mask.jpeg", out)

img_binary = (img > 127) * img  ## should be replaced by otsu thresholded image
cv2.imwrite(f"{img_name}_binary.jpeg", img_binary)

res = (out== 0) * img_binary
cv2.imwrite(f"{img_name}_reggrow_output.jpeg", res)