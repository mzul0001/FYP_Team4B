import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import feature

def HOG_FE(img):
    BGR_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dim = (width, height) = (240, 240)
    # resize image so hog does not miss cascading part of the image especially the edges
    BGR_img = cv2.resize(BGR_img, dsize=dim)

    # calculate HOG features
    kp, new_img = feature.hog(BGR_img,
                           orientations=9,  # Number of orientation bins. default is 9
                           pixels_per_cell=(8, 8),  # Size (in pixels) of a cell.
                           cells_per_block=(2, 2),  # Number of cells in each block.
                           visualize=True,
                           multichannel=False)  # If True, the last image dimension is considered as a color channel, otherwise as spatial.

    plt.imshow(new_img)
    plt.axis('off')
    plt.savefig('HOG_result.jpg')
    plt.show()

if __name__ == '__main__':
    img = cv2.imread("./inputs/"+'testcase 178.jpg', cv2.IMREAD_GRAYSCALE)
    HOG_FE(img)
    
    
