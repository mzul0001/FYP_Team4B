import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
from keras.layers import Flatten,Dense, Dropout
from nn import trainNN, saveModel


def createModel():
    '''
    This function is to create a structure of CNN model
    3 inputs layers, which outputs 16 or 32 feature maps which is
    created by sliding kernel from top left to bottom right
    precondition:
    :param:
    postcondition:
    :return: the CNN model
    error handling:
    '''
    # construct model
    # conv2D(filter -- number of convolution filters,kernel_size (n*n),
    # padding same -- output size is same with original
    # input shape -- the dimension of the input data)
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # 3rd, 4th, 5th, 6th layer
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # 7th, 8th layer
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.summary()

    # prediction layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(7, activation='softmax'))  ##number of emotions
    return model


def preprocessData(x_train):
    '''
    This function is to reshape,normalize and One-hot encode the input data so that keras can process them without
    errors.
    precondition:
    :param  x_train: array of facial image data
            y_train: array of labels of emotions
    postcondition:
    :return: it returns processed data
    '''
    print('Processing data...')
    # feature scalling (normalization) on image data
    # gray sclale is 1 channel so its range is between 0 and 255
    # min 0, max 1, RGB is 0 to 255, thus it divides by 255
    x_train = x_train.astype('float32')/255
    x_train = x_train.reshape((int(x_train.shape[0]),48,48,1))
    return x_train


def loadData(x_temp, y_temp, fileName, label):
    '''
    This function is to load data set from a folder into arrays
    precondition:
    :param  x_temp: array which is going to contain array of image data, 2-dimensional array
            y_temp: array which is going to contain labels
    postcondition:
    :return:
    error handling:
    '''
    print('Reading images...')
    imgs = glob.glob(fileName + '/*.' + 'jpg')
    for img in imgs:
        temp = img_to_array(load_img(img, target_size=(48, 48), color_mode='grayscale'))
        x_temp.append(temp)
        y_temp.append(label)


def prepareData(x_train,y_train,address):
    '''
    This function is to prepare data by calling loadData and preprocessData functions
    precondition:
    :param  x_train: array which is going to contain array of image data, 2-dimensional array
            y_train: array which is going to contain labels
            address: address of data set in github
    postcondition:
    :return: it returns data set which keras can already process
    '''
    loadData(x_train, y_train, address + 'angry', 0)
    loadData(x_train, y_train, address + 'disgust', 1)
    loadData(x_train, y_train, address + 'fear', 2)
    loadData(x_train, y_train, address + 'happy', 3)
    loadData(x_train, y_train, address + 'neutral', 4)
    loadData(x_train, y_train, address + 'sad', 5)
    loadData(x_train, y_train, address + 'surprise', 6)

    # convert it to numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # preprocess dataset so that keras can process them
    x_train = preprocessData(x_train)
    return x_train, y_train


def makeCNN():
    '''
    This function is to build a model and let the model study based on the given data set and save it at the end.
    :param  x_train: array which is going to contain array of image data, 2-dimensional array
            y_train: array which is going to contain labels
    :return:
    error handling:
    '''
    address = './face-expression-recognition-dataset/train/'
    x_train, y_train = prepareData([], [], address)
    address = './face-expression-recognition-dataset/validation/'
    x_test , y_test = prepareData([], [], address)
    print('Compiling convolutional neural network...')
    model = createModel()

    # categorical cross entropy -- also called multi class log loss
    #                           -- measures the performance of a classification
    #                              model where the prediction input is a probability value between 0 and 1
    # sparse -- the classes are labelled as consecutive numbers starting from 0
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model, x_train, x_test, y_train, y_test


def main():
    model, x_train, x_test, y_train, y_test = makeCNN()
    model = trainNN(model, x_train, x_test, y_train, y_test, 256, 30)
    # saveModel(model, 'model.json', 'model.h5')


if __name__ == '__main__':
    main()


    

