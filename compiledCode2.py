import cv2
import numpy as np


def IP(img, count):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Histogram Equalization
    gray = cv2.equalizeHist(gray)

    # Image Sharpening with Gaussian Blur
    gaussian = cv2.GaussianBlur(gray, (9, 9), 10.0)
    gray = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0, gray)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    color = (0, 0, 255)

    # scale factor -- how much the image size is reduced at each image scale
    # minNeighbours -- how many neighbors each candidate rectangle should have to retain it
    #               -- If the value is bigger, it detects less objects but less misdetections. If smaller,
    #               it detects more objects but more misdetections.
    # For each resulting detection, `levelWeights` will then contain the certainty of classification at the final stage.
    faces, _, levelWeights = cascade.detectMultiScale3(image=gray, scaleFactor=1.3, minNeighbors=6,
                                                       outputRejectLevels=True)
    # draw a rectangle for each face
    for (x, y, width, height) in faces:
        # image, left up coordinate, right down coordinate, color, thickness
        img = cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
        # crop image
        saved = img[y:y + height, x:x + width]
        count += 1

    return img, count


def faceDetection(filename):
    # read a video file
    video = cv2.VideoCapture(filename)
    count = 0  # count is the number of images detected and saved

    while video.isOpened():
        # read a frame from video
        ret, frame = video.read()

        if count == 0:
            height, width, layers = frame.shape
            size = (width,
                    height)
            # out = cv2.VideoWriter(filename='output.mp4', apiPreference=0, fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            # fps=15, frameSize=size)

        # if there is no next frame, the loop terminates
        if not ret: break

        cv2.waitKey(1)
        frame, count = IP(frame, count)  # detect a face in an image

        # out.write(frame)
        cv2.imshow('frame', frame)

        # stop its execution by pressing Q-key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

    # release memory
    video.release()
    cv2.destroyAllWindows()
    # out.release()
    return count


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json


def predict(features):
    # load json and create model
    with open('model.json', 'r') as json_file:
        loadedModel = model_from_json(json_file.read())

    # load weights into new model
    loadedModel.load_weights('model.h5')
    print("Loaded model")

    loadedModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # predict the class
    label = loadedModel.predict_classes(features)
    return label

def trainNN(model, train_features, test_features, train_labels, test_labels):
    print("Training NN...")
    # batch_size and epochs can be experimented to change accuracy
    trained = model.fit(train_features, train_labels, batch_size=256, epochs=20, verbose=1, validation_data=(
        test_features, test_labels))
    print("Evaluating NN...")
    [loss, accuracy] = model.evaluate(test_features, test_labels)

    # check for overfitting/underfitting
    # plot loss curves
    # plt.plot(trained.history['loss'], 'r')
    # plt.plot(trained.history['val_loss'], 'b')
    # plt.legend(['Training loss', 'Test loss'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve')
    # # plt.show()
    # print(trained.history.keys())
    # # plot accuracy curves
    # plt.plot(trained.history['accuracy'], 'r')
    # plt.plot(trained.history['val_accuracy'], 'b')
    # plt.legend(['Training accuracy', 'Test Accuracy'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy Curve')
    # plt.show()

    # serialize model to JSON
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights('model.h5')
    print("Saved model")

def buildNN(data):
    units = 10
    classes = 7
    data = data.shape[1]

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
    print("Compiled NN")
    return model


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
    return features


import os
from keras.utils import np_utils


def processDataset():
    address = './jaffe/'
    dirList = os.listdir(address)

    images = []

    for dir in dirList:
        imgList = os.listdir(address+dir)
        for img in imgList:
            imgRead = cv2.imread(address+dir+'/'+img)
            images += [cv2.resize(imgRead, (240, 240))]

    images = np.asarray(images)

    data = []
    for i in range(images.shape[0]):
        data += [featureExtraction_HOG(images[0])]
    data = np.asarray(data)

    dim = np.prod(data.shape[1:])
    data = data.reshape(data.shape[0], dim)
    data = data.astype('float32')
    data /= 255

    labels = np.array([0 for _ in range(data.shape[0])])
    labels[30:59] = 1
    labels[60:92] = 2
    labels[93:124] = 3
    labels[125:155] = 4
    labels[156:187] = 5
    labels[188:] = 6
    labels = np_utils.to_categorical(labels)

    seed = np.arange(data.shape[0])
    np.random.shuffle(seed)
    data = data[seed]
    labels = labels[seed]

    n = int(data.shape[0]*0.15)
    test_features = data[0:n]
    test_labels = labels[0:n]
    train_features = data[n:]
    train_labels = labels[n:]

    model = buildNN(data)
    trainNN(model, train_features, test_features, train_labels, test_labels)
    # category = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']
    return


def main():
    faceDetection('sample2a.mp4')
    # processDataset()


if __name__ == '__main__':
    main()
