import cv2
import numpy as np
import os

# Convert to grayscale
img = cv2.imread('lenna.png')

gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0, img)
cv2.imwrite("sharpened.jpg", sharpened)
