import cv2


def featureExtraction_HOG(img):
    BGR_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dim = (width, height) = (240, 240)
    # resize image so hog does not miss cascading part of the image especially the edges
    img = cv2.resize(img, dsize=dim)

    # Sobel: discrete differentiation operator
    # edges of an image is detected by obtaining the first derivative
    # horizontal edge
    # hy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    # vertical edge
    # hx = cv2.Sobel(img, cv2.CV_32F, 1, 0)

    # obtain second derivative
    # magnitude, direction = cv2.cartToPolar(hx, hy, angleInDegrees=True)

    # non-default HOG values
    winSize = (240, 240)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, 1, -1, 0, 0.2, 1, 64)
    features = hog.compute(img)
    return features
