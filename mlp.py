import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from extractFeatures import extractFeatures
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from sklearn.metrics import classification_report


def predict(features):
    '''
    function predicts the category of the input using a neural network model
    precondition:
    :param features: the model input
    postcondition:
    :return: the predicted category of the input
    '''
    print("Loading model...")
    # load json and create model
    with open('mlp_model.json', 'r') as json_file:
        loaded_model = model_from_json(json_file.read())

    # load weights into new model
    loaded_model.load_weights('mlp_model.h5')

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
    loaded_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # predict the class
    label = loaded_model.predict_classes(features)
    return label


def trainNN(model, train_features, test_features, train_labels, test_labels):
    '''
    function trains a neural network model
    precondition:
    :param  model: the neural network model
            train_features: the model input for training
            test_features: the model input for testing trained model
            train_labels: the model supposed output of each training input
            test_labels: the model supposed output of each testing input
    postcondition:
    :return:
    '''
    # train the neural network model
    # batch_size -- the number of data fed into the model at each iteration
    # epochs -- the number of repetition to feed all data into the model
    # verbose -- training overview display method
    # validation_data -- the testing data set
    # the number of iterations per epoch = total data/batch_size
    # the total number of iterations = the number of iterations per epoch * epoch
    trained = model.fit(train_features, train_labels, batch_size=512, epochs=8000, verbose=2, validation_data=(
        test_features, test_labels), shuffle=True)
    print("Evaluating NN...")
    [loss, accuracy] = model.evaluate(test_features, test_labels)
    print("Loss = {}, accuracy = {}".format(loss, accuracy))
    predicted = model.predict_classes(test_features)
    category = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']
    print(classification_report(test_labels, predicted, target_names=category))

    # # check for overfitting/underfitting
    # # loss curve
    # plt.plot(trained.history['loss'], 'r')
    # plt.plot(trained.history['val_loss'], 'b')
    # plt.legend(['Training loss', 'Test loss'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve')
    # plt.show()
    #
    # # accuracy curve
    # plt.plot(trained.history['accuracy'], 'r')
    # plt.plot(trained.history['val_accuracy'], 'b')
    # plt.legend(['Training accuracy', 'Test Accuracy'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy Curve')
    # plt.show()

    # serialize model to JSON
    model_json = model.to_json()
    with open('mlp_model.json', 'w') as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights('mlp_model.h5')
    return


def buildNN(data, units=832, classes=7):
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
    for i in range(2):
        # activation -- the function used by the neurons for calculation
        model.add(Dense(832, activation='relu'))
        model.add(Dropout(0.5))

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


def processDataset():
    '''
    function loads images from different classes to arrays; one for training and one for testing
    precondition: the directory storing the images exist
    :param:
    postcondition: the images in the directory is not modified
    :return:
    '''
    address = './face-expression-recognition-dataset/'
    dir_list = os.listdir(address)

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    print("Reading images...")
    for dir in dir_list:
        folder_list = os.listdir(address + dir)
        count = 0
        for folder in folder_list:
            img_list = os.listdir(address + dir + '/' + folder)
            for img in img_list:
                img_read = cv2.imread(address + dir + '/' + folder + '/' + img)
                if dir == 'train':
                    train_images += [img_read]
                    train_labels += [count]
                elif dir == 'validation':
                    test_images += [img_read]
                    test_labels += [count]
            count += 1
    train_images = np.asarray(train_images)
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
    trainNN(model, train_features, test_features, train_labels, test_labels)


if __name__ == '__main__':
    main()
