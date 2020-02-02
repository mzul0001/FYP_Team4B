import numpy as np
import os
import cv2
from extractFeatures import extractFeatures
from nn import trainNN, saveModel
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers


def buildNN(data, classes=7):
    '''
    function creates a multi layer perceptron model
    precondition: the input shape condition of the model is satisfied
    :param  data: the model input
            units: the number of neurons in each hidden layer
            classes: the number of prediction category to be made
    postcondition:
    :return: the multi layer perceptron model
    '''
    input_shape = data.shape[1]

    model = Sequential()
    # Dropout layer -- stops n% of neurons in the hidden layers from functioning
    #               -- to avoid overfitting
    model.add(Dropout(0.5, input_shape=(input_shape,)))
    # hidden layers
    # activation -- the function used by the neurons for calculation
    model.add(Dense(832, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(640, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(832, activation='relu'))
    model.add(Dropout(0.2))

    # output layer
    model.add(Dense(classes, activation='softmax'))

    # Adam (Adaptive Moment Estimation) -- an algorithm for first-order gradient-based optimization of stochastic
    #                                      objective functions, based on adaptive estimates of lower-order moments
    #                                   -- computationally efficient, has little memory requirements
    #                                   -- suited for problems that are large in terms of data and/or parameters
    # learning_rate -- Adam's default learning rate is 0.001
    #               -- reduced to avoid overfitting while attempting to increase accuracy
    optimizer = optimizers.Adam(learning_rate=0.0008)

    # categorical cross entropy -- also called multi class log loss
    #                           -- measures the performance of a classification
    #                              model where the prediction input is a probability value between 0 and 1
    # sparse -- the classes are labelled as consecutive numbers starting from 0
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def loadImages(address, label, img_array, label_array):
    img_list = os.listdir(address)
    for img in img_list:
        img_read = cv2.imread(address+img)
        img_array += [img_read]
        label_array += [label]
    return img_array, label_array


def processDataset():
    '''
    function loads images from different classes to arrays; one for training and one for testing
    precondition: the directory storing the images exist
    :param:
    postcondition: the images in the directory is not modified
    :return:
    '''
    address = './face-expression-recognition-dataset/'

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    print("Reading images...")
    loadImages(address + 'train/angry/', 0, train_images, train_labels)
    loadImages(address + 'train/disgust/', 1, train_images, train_labels)
    loadImages(address + 'train/fear/', 2, train_images, train_labels)
    loadImages(address + 'train/happy/', 3, train_images, train_labels)
    loadImages(address + 'train/neutral/', 4, train_images, train_labels)
    loadImages(address + 'train/sad/', 5, train_images, train_labels)
    loadImages(address + 'train/surprise/', 6, train_images, train_labels)
    train_images = np.asarray(train_images)

    loadImages(address + 'validation/angry/', 0, test_images, test_labels)
    loadImages(address + 'validation/disgust/', 1, test_images, test_labels)
    loadImages(address + 'validation/fear/', 2, test_images, test_labels)
    loadImages(address + 'validation/happy/', 3, test_images, test_labels)
    loadImages(address + 'validation/neutral/', 4, test_images, test_labels)
    loadImages(address + 'validation/sad/', 5, test_images, test_labels)
    loadImages(address + 'validation/surprise/', 6, test_images, test_labels)
    test_images = np.asarray(test_images)

    print("Extracting features...")
    train_features = []
    for i in range(train_images.shape[0]):
        train_features += [extractFeatures(train_images[i])]
    train_features = np.asarray(train_features)

    test_features = []
    for i in range(test_images.shape[0]):
        test_features += [extractFeatures(test_images[i])]
    test_features = np.asarray(test_features)

    print('Processing data...')
    dim = np.prod(train_features.shape[1:])
    train_features = train_features.reshape(train_features.shape[0], dim)
    train_features = train_features.astype('float32')
    train_features /= 255

    test_features = test_features.reshape(test_features.shape[0], dim)
    test_features = test_features.astype('float32')
    test_features /= 255

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    seed = np.arange(train_features.shape[0])
    np.random.shuffle(seed)
    train_features = train_features[seed]
    train_labels = train_labels[seed]

    seed = np.arange(test_features.shape[0])
    np.random.shuffle(seed)
    test_features = test_features[seed]
    test_labels = test_labels[seed]
    return train_features, test_features, train_labels, test_labels


def main():
    train_features, test_features, train_labels, test_labels = processDataset()
    print('Compiling multi layer perceptron model...')
    model = buildNN(train_features)
    model = trainNN(model, train_features, test_features, train_labels, test_labels, 512, 600)
    # saveModel(model, 'mlp_model.json', 'mlp_model.h5')


if __name__ == '__main__':
    main()
