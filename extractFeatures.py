import cv2
import numpy as np
import matplotlib.pyplot as plt


def extractFeatures(img):
    '''
    function compute the hog features of an image
    precondition:
    :param img: the image to compute the hog features on
    postcondition: the original image is not modified
    :return: the hog features of the image
    '''
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dim = (48, 48)
    # resize image so hog sliding window does not go out of the image bounds
    gray_img = cv2.resize(gray_img, dsize=dim)
    # non-default HOG values
    win_size = dim
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = cell_size
    nbins = 9

    # win_size -- size of image
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, 1, -1, 0, 0.2, 1, 64)
    features = hog.compute(gray_img)

    # # reference: https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python
    # n_cells = (gray_img.shape[0] // cell_size[0], gray_img.shape[1] // cell_size[1])
    # n_inner = (block_size[0] // cell_size[0], block_size[1] // cell_size[1])
    #
    # features = features.reshape(n_cells[1] - n_inner[1] + 1, n_cells[0] - n_inner[0] + 1, n_inner[1], n_inner[0], nbins)
    # features = features.transpose((1, 0, 2, 3, 4))
    #
    # gradients = np.zeros((n_cells[0], n_cells[1], nbins))
    #
    # # count cells (border cells appear less often across overlapping groups)
    # cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)
    #
    # for y in range(n_inner[0]):
    #     for x in range(n_inner[1]):
    #         gradients[y:n_cells[0] - n_inner[0] + y + 1, x:n_cells[1] - n_inner[1] + x + 1] += features[:, :, y, x, :]
    #         cell_count[y:n_cells[0] - n_inner[0] + y + 1, x:n_cells[1] - n_inner[1] + x + 1] += 1
    #
    # # average gradients
    # gradients /= cell_count
    #
    # bin = 2  # angle is 360 / nbins * direction
    # plt.pcolor(gradients[:, :, bin])
    # plt.gca().invert_yaxis()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.colorbar()
    # plt.show()
    return features


if __name__ == '__main__':
    img = cv2.imread('./jaffe/ANGRY/angry1.jpg')
    extractFeatures(img)