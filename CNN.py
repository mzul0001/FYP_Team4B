import keras
import glob

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array,load_img

from keras.layers import Flatten,Dense, Dropout,Activation, BatchNormalization
from keras.optimizers import Adam

from keras.models import model_from_json
from sklearn.metrics import classification_report



def LoadData(x_temp,y_temp,fileName,label):
    '''
        Functionality of the function
        This function is to load dataset from a folder into arrays

        Error handle: there is no error handle
        Return: no returns
        Parameter x_temp: array which is going to contain array of image data, 2-dimentional array
        Parameter y_temp: array which is going to contain labels

    '''
    Imgs = glob.glob(fileName+'/*.'+'jpg')
    for img in Imgs:
        
        temp = img_to_array( load_img(img,target_size = (48,48),color_mode='grayscale') )
        x_temp.append(temp)
        y_temp.append(label)

def PreprocessData(x_train,y_train):
    '''
        Functionality of the function
        This function is to reshape,normalize and One-hot encode the input data so that keras can process them without errors.

        Error handle: there is no error handle
        Return: it returns processed data
        Parameter x_train: array of facial image data
        Parameter y_train: array of labels of emotions

    '''
    #feature scalling (normalization) on image data
    #gray sclale is 1 channel so its range is between 0 and 255
    ##min 0 max 1, RGB is 0 to 255, thus it devides by 255
    x_train = x_train.astype('float32') /255
    
    ##keras data type is (batch size,width,height,channel)
    
    x_train = x_train.reshape((int(x_train.shape[0]),48,48,1))
    
    

    ##label is One-hot encoded ---all avlues are 1 or 0 which avoids errors in learning
    y_train = keras.utils.to_categorical(y_train)  
    

    return x_train,y_train


        

def PrepareData(x_train,y_train,address):
    '''
        Functionality of the function
        This function is to prepare data by calling loadData and PreprocessData functions

        Error handle: there is no error handle
        Return: it returns dataset which keras can already process
        Parameter x_train: array which is going to contain array of image data, 2-dimentional array
        Parameter y_train: array which is going to contain labels
        Parameter address: address of dataset in laptop

    '''
    #------load data x is image, y is label (ex 1,5,3...)  -------------------
    ##(x_train, y_train), (x_test, y_test) = mnist.load_data()
    ##0 angry,
    ##1 disgust,
    ##2 fear,
    ##3 happy,
    ##4 sad,
    ##5 surprise,
    ##6 neutral    

    
    
    LoadData(x_train,y_train,address +'angry',0)
    LoadData(x_train,y_train,address +'disgust',1)
    LoadData(x_train,y_train,address +'fear',2)
    LoadData(x_train,y_train,address +'happy',3)
    LoadData(x_train,y_train,address +'sad',4)
    LoadData(x_train,y_train,address +'surprise',5)
    LoadData(x_train,y_train,address +'neutral',6)


    #convert it to numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    size = int(x_train.shape[0]) /(48*48)
    
    ##print('X_train:', x_train.shape, 'y_train:', y_train.shape)

    #---------finished loading dataset-----------------------------------------

    #preprocess dataset so that keras can process them
    x_train,y_train = PreprocessData(x_train,y_train)
    return x_train,y_train



def CreateModel():
    '''
        Functionality of the function
        This function is to create a structure of CNN model
        3 inputs layers, which outputs 16 or 32 feature maps which is created by sliding kernel from top left to bottom right

        Maxpooling reduce the size of feature map taking max values of each region with 2*2 filters of stride 2.
        It is to downsample the input img and reduce computational costs, which prevents over-fitting
        by extracting abstract form of features by only taking some parameters.

        Dropout layer is set some parameters zero randomely based on given percentage so that it reduce bias made from training dataset.
        
        Flatten layer converts the output of the convolutional layers to vector of one dimentional array which size is 64.

        Dense layer(Output layer) outputs probabilities for each class using softmax function that guarantees output between 0 and 1.

        Error handle: there is no error handle
        Return: it returns CNN model

    '''
    
    #construct model
    ##conv2D(filter--number of convolution filters,kernel_size (n*n), padding same--output size is same with original input size,
    ##input data type)
    model = Sequential()

    #first layer
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1', input_shape=(48,48,1)) )
    model.add(MaxPooling2D(pool_size=(2,2)) )
    model.add(Dropout(0.3))
    #model.add(BatchNormalization())

    #Second, third layer
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2', input_shape=(48,48,1)) )
    model.add(Dropout(0.3))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3', input_shape=(48,48,1)) )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    #model.add(BatchNormalization())

    #4th,5th,6th layer
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4', input_shape=(48,48,1)) )
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv5', input_shape=(48,48,1)) )
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv6', input_shape=(48,48,1)) )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    #model.add(BatchNormalization())

    #7th,8th,9th layer
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv7', input_shape=(48,48,1)) )
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv8', input_shape=(48,48,1)) )
    model.add(MaxPooling2D(pool_size=(2,2)) )
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv9', input_shape=(48, 48, 1)))
    model.add(Dropout(0.3))
    #model.add(Conv2D(64, (4, 4), activation='relu', padding='same', name='conv10', input_shape=(48, 48, 1)))
    #model.add(BatchNormalization())

    
    
    model.summary()

    ##prediction layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, activation='softmax'))##number of emotions
    return model



def SaveModel(model):
    '''
        Functionality of the function
        This function is to save a model and its weight as jason file and h5 file

        Error handle: there is no error handle
        Return: no returns
        Parameter model: compiled and studied model 

    '''
    model_json = model.to_json()
    with open('model.json','w') as json_file:
        json_file.write(model_json)

    model.save_weights('model.h5')
    print("Saved model")



def CNN_make(x_train,y_train):
    '''
        Functionality of the function
        This function is to build a model and let the model study based on the given dataset and save it at the end.

        Error handle: there is no error handle
        Return: no returns
        Parameter x_train: array which is going to contain array of image data, 2-dimentional array
        Parameter y_train: array which is going to contain labels

    '''
    address = './img_recognition/train/'
    x_train,y_train = PrepareData(x_train,y_train,address)
    address = './img_recognition/validation/'
    x_test =[]
    y_test =[]
    x_test,y_test = PrepareData(x_test,y_test,address)
    
    model = CreateModel()

    #compile  categorical_crossentropy is used when label is one-hot encoded and it is N-classes classification
    #adam 
    model.compile(loss= 'categorical_crossentropy',optimizer ='adam' , metrics=['accuracy'])

    #learning
    #batch size is the number of samples that will be passed to neural network per iteration until all samples are propagated.
    #less batch size uses less memory.

    #epoch is how many times you go through all samples. every epoch it tries shuffled samples
    history =model.fit(x_train,y_train,batch_size= 128,epochs= 200,validation_data=(x_test,y_test))
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    #save model
    SaveModel(model)


   
    

def CNN_predict(img,SavedModel):
    '''
        Functionality of the function
        This function is to predict corresponding class(or emotion) to an input of image

        Error handle: there is no error handle
        Return: it returns string of predicted emotion 
        Parameter img: facial image
        Parameter SavedModel: trained model to predict a class.

    '''

    x_test = []
    
    temp = img_to_array(img)

    x_test.append(temp)
    x_test = np.array(x_test)
    x_test = x_test.astype('float32') /255
    
    label = SavedModel.predict_classes(x_test)

    
    item = label
    if item ==0:
        return "angry"
    elif item == 1:
        return "disgust"
    elif item == 2:
        return "fear"
    elif item == 3:
        return "happy"
    elif item == 4:
        return "sad"
    elif item == 5:
        return "surprise"
    elif item == 6:
        return "neutral"
    
  




def CNN_evaluate(x_train,y_train):
    '''
        Functionality of the function
        This function is to evaluate model.It gives precision,recall,f1 score, support, accuracy and loss.

        Error handle: there is no error handle
        Return: no returns
        Parameter x_train: facial images which will be used as testing data
        Parameter y_train: labels of emotions which will be used as testing data

    '''
    address = './img_recognition/validation/'
    x_test = []
    y_test = []
    x_test,y_test = PrepareData(x_train,y_train,address)


    with open('model.json','r') as json_file:
        SavedModel = model_from_json(json_file.read())
    SavedModel.load_weights('model.h5')
    SavedModel.compile(loss= 'categorical_crossentropy',optimizer ='adam' , metrics=['accuracy'])

    y_pred = SavedModel.predict(x_test, batch_size=64, verbose=0)
    y_pred_bool = np.argmax(y_pred, axis=1)
    y_test_bool = np.argmax(y_test, axis=1)
    print()
    print("0:angry 1:disgust 2:fear 3:happy 4:sad 5:surprise 6:neutral")
    print(classification_report(y_test_bool, y_pred_bool))
    
    score = SavedModel.evaluate(x_test, y_test, batch_size= 64,verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

   

if __name__ == '__main__':

    x_train =[]
    y_train =[]
    x_test = []
    y_test = []
    
    #newly create model
    CNN_make(x_train,y_train)

    #load model and
    

    CNN_evaluate(x_test,y_test)
    


    

