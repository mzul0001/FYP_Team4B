import cv2
import numpy as np

def IP(img):
    address = 'C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages\\cv2\\data\\' \
              'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(address)

    # convert RGB image to BGR
    BGR_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(BGR_img, 1.3, 6)

    for (x, y, width, height) in faces:
        # crop image
        croppedFace = img[y:y + height, x:x + width]
        train_features = featureExtraction_HOG(croppedFace).flatten()
        color = (0, 0, 255)
        # draw rectangle around the face on the image
        img = cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
    return img


def faceDetection(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()
        # if there's no frame, return value (ret) is false
        if not ret: return False

        frame = frame.transpose(0, 1, 2)
        # resize the frame to reduce computation
        # frame = cv2.resize(frame, dsize=None, fx=0.7, fy=0.7)
        # process frame
        frame = IP(frame)

        cv2.imshow('output', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): return False

    video.release()
    cv2.destroyAllWindows()
    return True


from keras.models import Sequential
from keras.layers import Dense

def trainNN(model, train_features, test_features, train_labels, test_labels):
    # batch_size and epochs can be experimented to change accuracy
    trained = model.fit(train_features, train_labels, batch_size=256, epochs=20, verbose=1, validation_data=(
        test_features, test_labels))
    [loss, accuracy] = model.evaluate(test_features, test_labels)


def buildNN(train_features):
    units = 10
    classes = 7
    data = train_features.shape[1]

    model = Sequential()
    for i in range(10):
        # hidden layers
        model.add(Dense(units, input_shape=(data,), activation='relu'))
    # output layer
    model.add(Dense(classes, input_shape=(data,), activation='softmax'))

    # configure model
    #
    # Adam (Adaptive Moment Estimation):
    # - an algorithm for first-order gradient-based optimization of stochastic objective functions,
    # based on adaptive estimates of lower-order moments
    # - computationally efficient, has little memory requirements
    # - suited for problems that are large in terms of data and/or parameters
    #
    # categorical cross entropy: - also called multi class log loss - measures the performance of a classification
    # model where the prediction input is a probability value between 0 and 1

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


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


def main():
    faceDetection('sample2b.mp4')


if __name__ == '__main__':
    main()
