# -*- coding: utf-8 -*-

import cv2
import numpy as np

image = cv2.imread('bacteria.tif',0)
print(image.shape)

sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)


N = image.shape[0] #row 
M = image.shape[1] #column


sobelxImage = np.zeros((N,M))
sobelyImage = np.zeros((N,M))
sobelGrad = np.zeros((N,M))

#Surrounds array with 0's on the outside perimeter
image = np.pad(image, ((1,1),(1,1)), 'constant')
print(image.shape)
 
for i in range(N):
    for j in range(M):        
        gx = (sobelx[0][0] * image[i][j]) + \
             (sobelx[0][1] * image[i][j+1]) + \
             (sobelx[0][2] * image[i][j+2]) + \
             (sobelx[1][0] * image[i+1][j]) + \
             (sobelx[1][1] * image[i+1][j+1]) + \
             (sobelx[1][2] * image[i+1][j+2]) + \
             (sobelx[2][0] * image[i+2][j]) + \
             (sobelx[2][1] * image[i+2][j+1]) + \
             (sobelx[2][2] * image[i+2][j+2])

        gy = (sobely[0][0] * image[i][j]) + \
             (sobely[0][1] * image[i][j+1]) + \
             (sobely[0][2] * image[i][j+2]) + \
             (sobely[1][0] * image[i+1][j]) + \
             (sobely[1][1] * image[i+1][j+1]) + \
             (sobely[1][2] * image[i+1][j+2]) + \
             (sobely[2][0] * image[i+2][j]) + \
             (sobely[2][1] * image[i+2][j+1]) + \
             (sobely[2][2] * image[i+2][j+2])     

        sobelxImage[i][j] = gx
        sobelyImage[i][j] = gy


        g = np.sqrt(gx * gx + gy * gy)
        sobelGrad[i][j] = g


cv2.imwrite('custom_2d_convolution_gx.png',sobelxImage) 
cv2.imwrite('custom_2d_convolution_gy.png',sobelyImage)
cv2.imwrite('custom_2d_convolution_gradient.png',sobelGrad)

