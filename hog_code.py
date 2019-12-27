import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import feature

def HOG_FE(img):
    ##gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#convert it into gray
    ##hog = cv2.HOGDescriptor()
    ##kp = hog.compute(gray,None)

    
    # calculate HOG features
    kp,newimg = feature.hog(img,
                           orientations=9,#Number of orientation bins. default is 9
                           pixels_per_cell=(8, 8),#Size (in pixels) of a cell.
                           cells_per_block=(1, 1),#Number of cells in each block.
                           visualize=True,
                           multichannel=False)#If True, the last image dimension is considered as a color channel, otherwise as spatial.

    plt.imshow(newimg)
    plt.axis('off')
    plt.savefig('HOG_result.jpg')
    plt.show()

if __name__ == '__main__':
    img = cv2.imread("./inputs/"+'testcase 178.jpg', cv2.IMREAD_GRAYSCALE)
    HOG_FE(img)
    
    
