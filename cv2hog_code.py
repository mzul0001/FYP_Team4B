import cv2


def featureExtraction_HOG(img):
    BGR_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dim = (width, height) = (240, 240)
    # resize image so hog does not miss cascading part of the image especially the edges
    BGR_img = cv2.resize(BGR_img, dsize=dim)

    # Sobel: discrete differentiation operator
    # edges of an image is detected by obtaining the first derivative
    # horizontal edge
    # hy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    # vertical edge
    # hx = cv2.Sobel(img, cv2.CV_32F, 1, 0)

    # obtain second derivative
    # magnitude, direction = cv2.cartToPolar(hx, hy, angleInDegrees=True)

    # non-default HOG values
    winSize = (BGR_img.shape[0], BGR_img[1])
    cellSize = (8, 8)
    blockSize = (cellSize[0]+cellSize[0], cellSize[1]+cellSize[1])
    blockStride = (cellSize[0], cellSize[1])
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, 1, -1, 0, 0.2, 1, 64)
    features = hog.compute(BGR_img)
    return features