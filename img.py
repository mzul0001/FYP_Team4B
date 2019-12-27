import cv2
import numpy as np
import os

# Convert to grayscale
img = cv2.imread('lena.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Histogram Equalization
equ = cv2.equalizeHist(img)

gaussianBlur = equ.copy()

height = equ.shape[0]
width = equ.shape[1]

# Gaussian kernal
gaussian = (1.0/57) * np.array(
        [[0, 1, 2, 1, 0],
        [1, 3, 5, 3, 1],
        [2, 5, 9, 5, 2],
        [1, 3, 5, 3, 1],
        [0, 1, 2, 1, 0]])
sum(sum(gaussian))

for i in np.arange(2, height-2):
    for j in np.arange(2, width-2):
        sum = 0
        for k in np.arange(-2, 3):
            for l in np.arange(-2, 3):
                a = equ.item(i+k, j+l)
                p = gaussian[2+k, 2+l]
                sum = sum + (p * a)
        b = sum
        gaussianBlur.itemset((i,j), b)

cv2.imwrite('GaussianBlur.png',gaussianBlur)

# unsharp filtering method
c = equ - gaussianBlur
sharpened = equ + c
cv2.imwrite('sharpened.png',sharpened)

# displaying enhancements
res = np.hstack((img, equ, sharpened)) #stacking images side-by-side
cv2.imwrite('res.png',res)

# deleting files
os.remove('GaussianBlur.png')
#remove original pic





