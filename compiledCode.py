import cv2
import numpy as np


def ImageProcessing(img, imgName, faceNo):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Histogram Equalization
    gray = cv2.equalizeHist(gray)

    # Image Sharpening with Gaussian Blur
    gaussian = cv2.GaussianBlur(gray, (9, 9), 10.0)
    gray = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0, gray)

    # obtain cascade classifiers
    # FaceAddress ='C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python37-32\\'
    # FaceAddress +='Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
    # FaceCascade = cv2.CascadeClassifier(FaceAddress)

    # EyeAddress ='C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python37-32\\'
    # EyeAddress +='Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml'
    # EyeCascade = cv2.CascadeClassifier(EyeAddress)
    #
    # MouthAddress ='C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python37-32\\'
    # MouthAddress +='Lib\\site-packages\\cv2\\data\\haarcascade_mcs_mouth.xml'
    # MouthCascade = cv2.CascadeClassifier(MouthAddress)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    color = (0, 0, 255)
    # color2 = (0, 255, 0)
    # color3 = (0, 0, 0)

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
        cropped = img[y:y + height, x:x + width]
        # name of the image
        imgName += str(faceNo) + '.jpg'
        # save the image into the outputs file
        cv2.imwrite("./outputs/" + imgName, cropped)
        faceNo += 1
    return img, faceNo


def videoProcessing(videoName):
    # read a video file
    video = cv2.VideoCapture(videoName)
    # count is the number of images detected and saved
    count = 0
    # number of frames per second
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(filename='test_output.avi', apiPreference=0, fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                             fps=fps, frameSize=(width, height))

    while video.isOpened():
        # read a frame from video
        ret, frame = video.read()

        # if there is no next frame, the loop terminates
        if not ret: break

        current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        timestamp = current_frame / fps
        imgName = str(timestamp) + '-' + str(current_frame) + '-'
        # detect a face in an image
        frame, count = ImageProcessing(frame, imgName, count)
        writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

    # release memory
    video.release()
    writer.release()
    cv2.destroyAllWindows()
    return count


import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

class ModelMetrics(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        y_pred = np.asarray(self.model.predict_classes(self.validation_data[0])).round()
        y_target = self.validation_data[1]
        y_f1 = f1_score(y_target, y_pred, average='micro')
        y_recall = recall_score(y_target, y_pred, average='micro')
        y_precision = precision_score(y_target, y_pred, average='micro')
        self.f1_scores.append(y_f1)
        self.recalls.append(y_recall)
        self.precisions.append(y_precision)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import model_from_json
import matplotlib.pyplot as plt


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
    metrics = ModelMetrics()
    # batch_size and epochs can be experimented to change accuracy
    trained = model.fit(train_features, train_labels, batch_size=71, epochs=20, verbose=2, validation_data=(
        test_features, test_labels), callbacks=[metrics])
    print("Evaluating NN...")
    [loss, accuracy] = model.evaluate(test_features, test_labels)
    print("Loss = {}, accuracy = {}".format(loss, accuracy))
    predicted = model.predict_classes(test_features)
    category = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']
    print(classification_report(test_labels, predicted, target_names=category))

    # # check for overfitting/underfitting
    # # plot loss curves
    # plt.plot(trained.history['loss'], 'r')
    # plt.plot(trained.history['val_loss'], 'b')
    # plt.legend(['Training loss', 'Test loss'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve')
    # plt.show()
    #
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
    with open('model1.json', 'w') as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights('model1.h5')
    # print("Saved model")


def buildNN(data):
    units = 10
    classes = 7
    input = data.shape[1]

    model = Sequential()
    for i in range(units):
        # hidden layers
        model.add(Dense(units, input_shape=(input,), activation='relu'))
        # model.add(Dropout(0.2))
    # output layer
    model.add(Dense(classes, input_shape=(input,), activation='softmax'))

    # configure model
    #
    # Adam (Adaptive Moment Estimation):
    # - an algorithm for first-order gradient-based optimization of stochastic objective functions,
    # based on adaptive estimates of lower-order moments
    # - computationally efficient, has little memory requirements
    # - suited for problems that are large in terms of data and/or parameters
    #
    # categorical cross entropy:
    # - also called multi class log loss - measures the performance of a classification model where
    # the prediction input is a probability value between 0 and 1

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Compiled NN")
    return model


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
    winSize = dim
    cellSize = (8, 8)
    blockSize = (16, 16)
    blockStride = cellSize
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, 1, -1, 0, 0.2, 1, 64)
    features = hog.compute(BGR_img)

    # # reference: https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python
    # n_cells = (BGR_img.shape[0] // cellSize[0], BGR_img.shape[1] // cellSize[1])
    # n_inner = (blockSize[0] // cellSize[0], blockSize[1] // cellSize[1])
    # features = features.reshape(n_cells[1] - n_inner[1] + 1, n_cells[0] - n_inner[0] + 1, n_inner[1], n_inner[0], nbins)
    # features = features.transpose((1, 0, 2, 3, 4))
    #
    # gradients = np.zeros((n_cells[0], n_cells[1], nbins))
    #
    # # count cells (border cells appear less often across overlapping groups)
    # cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)
    #
    # for off_y in range(n_inner[0]):
    #     for off_x in range(n_inner[1]):
    #         gradients[off_y:n_cells[0] - n_inner[0] + off_y + 1,
    #         off_x:n_cells[1] - n_inner[1] + off_x + 1] += features[:, :, off_y, off_x, :]
    #         cell_count[off_y:n_cells[0] - n_inner[0] + off_y + 1,
    #         off_x:n_cells[1] - n_inner[1] + off_x + 1] += 1
    #
    # # Average gradients
    # gradients /= cell_count
    #
    # bin = 2  # angle is 360 / nbins * direction
    # plt.pcolor(gradients[:, :, bin])
    # plt.gca().invert_yaxis()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.colorbar()
    # plt.show()

    return features


import os
from keras.utils import np_utils


def processDataset_jaffe():
    address = './jaffe/'
    dirList = os.listdir(address)

    images = []

    print("Reading images...")
    for dir in dirList:
        imgList = os.listdir(address+dir)
        for img in imgList:
            imgRead = cv2.imread(address+dir+'/'+img)
            images += [imgRead]

    images = np.asarray(images)

    print("Extracting features...")
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
    # labels = np_utils.to_categorical(labels)

    seed = np.arange(data.shape[0])
    np.random.shuffle(seed)
    data = data[seed]
    labels = labels[seed]

    n = int(data.shape[0]*0.15)
    test_features = data[0:n]
    test_labels = labels[0:n]
    train_features = data[n:]
    train_labels = labels[n:]
    print("Processed images")

    model = buildNN(data)
    trainNN(model, train_features, test_features, train_labels, test_labels)
    return


def processDataset():
    address = './face-expression-recognition-dataset/images/'
    dirList = os.listdir(address)

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    print("Reading images...")
    for dir in dirList:
        if dir == 'train':
            count = 0
            folderList = os.listdir(address+dir)
            for folder in folderList:
                imgList = os.listdir(address+dir+'/'+folder)
                for img in imgList:
                    imgRead = cv2.imread(address+dir+'/'+folder+'/'+img)
                    train_images += [cv2.resize(imgRead, (48, 48))]
                    train_labels += [count]
                count += 1
    train_images = np.asarray(train_images)

    for dir in dirList:
        if dir == 'validation':
            count = 0
            folderList = os.listdir(address+dir)
            for folder in folderList:
                imgList = os.listdir(address+dir+'/'+folder)
                for img in imgList:
                    imgRead = cv2.imread(address+dir+'/'+folder+'/'+img)
                    test_images += [cv2.resize(imgRead, (48, 48))]
                    test_labels += [count]
                count += 1
    test_images = np.asarray(test_images)

    print("Extracting features...")
    train_data = []
    for i in range(train_images.shape[0]):
        train_data += [featureExtraction_HOG(train_images[0])]
    train_data = np.asarray(train_data)

    test_data = []
    for i in range(test_images.shape[0]):
        test_data += [featureExtraction_HOG(test_images[0])]
    test_data = np.asarray(test_data)

    dim = np.prod(train_data.shape[1:])
    train_data = train_data.reshape(train_data.shape[0], dim)
    train_data = train_data.astype('float32')
    train_data /= 255

    dim = np.prod(test_data.shape[1:])
    test_data = test_data.reshape(test_data.shape[0], dim)
    test_data = test_data.astype('float32')
    test_data /= 255

    train_labels = np.asarray(train_labels)
    # train_labels = np_utils.to_categorical(train_labels)

    test_labels = np.asarray(test_labels)
    # test_labels = np_utils.to_categorical(test_labels)

    seed = np.arange(train_data.shape[0])
    np.random.shuffle(seed)
    train_data = train_data[seed]
    train_labels = train_labels[seed]

    seed = np.arange(test_data.shape[0])
    np.random.shuffle(seed)
    test_data = test_data[seed]
    test_labels = test_labels[seed]
    print("Processed images")

    model = buildNN(train_data)
    trainNN(model, train_data, test_data, train_labels, test_labels)
    return


def main():
    # videoProcessing('sample2a.mp4')
    processDataset_jaffe()


if __name__ == '__main__':
    main()
