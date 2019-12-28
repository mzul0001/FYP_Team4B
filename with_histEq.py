import cv2
import numpy as np
import os

# Convert to grayscale
img = cv2.imread('lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Histogram Equalization
equ = cv2.equalizeHist(img)
cv2.imwrite("histEQ.jpg", equ)

gaussian = cv2.GaussianBlur(equ, (9, 9), 10.0)
sharpened = cv2.addWeighted(equ, 1.5, gaussian, -0.5, 0, equ)
cv2.imwrite("sharpened.jpg", sharpened)

# displaying enhancements
res = np.hstack((img, equ ,gaussian, sharpened)) #stacking images side-by-side
cv2.imwrite('res.png',res)