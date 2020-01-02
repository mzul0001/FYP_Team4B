import cv2
import numpy as np
import matplotlib.pyplot as plt


def featureExtraction_HOG(img):
    BGR_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dim = (width, height) = (240, 240)
    # resize image so hog does not miss cascading part of the image especially the edges
    BGR_img = cv2.resize(BGR_img, dsize=dim)

    # non-default HOG values
    winSize = (240, 240)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, 1, -1, 0, 0.2, 1, 64)
    features = hog.compute(BGR_img)

    # reference: https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python
    n_cells = (BGR_img.shape[0] // cellSize[0], BGR_img.shape[1] // cellSize[1])
    n_inner = (blockSize[0] // cellSize[0], blockSize[1] // cellSize[1])
    features = features.reshape(n_cells[1] - n_inner[1] + 1, n_cells[0] - n_inner[0] + 1, n_inner[1], n_inner[0], nbins)
    features = features.transpose((1, 0, 2, 3, 4))

    gradients = np.zeros((n_cells[0], n_cells[1], nbins))

    # count cells (border cells appear less often across overlapping groups)
    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

    for off_y in range(n_inner[0]):
        for off_x in range(n_inner[1]):
            gradients[off_y:n_cells[0] - n_inner[0] + off_y + 1,
            off_x:n_cells[1] - n_inner[1] + off_x + 1] += features[:, :, off_y, off_x, :]
            cell_count[off_y:n_cells[0] - n_inner[0] + off_y + 1,
            off_x:n_cells[1] - n_inner[1] + off_x + 1] += 1

    # Average gradients
    gradients /= cell_count

    bin = 2  # angle is 360 / nbins * direction
    plt.pcolor(gradients[:, :, bin])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.show()

    return features


if __name__ == '__main__':
    img = cv2.imread('./jaffe/ANGRY/angry1.jpg')
    featureExtraction_HOG(img)