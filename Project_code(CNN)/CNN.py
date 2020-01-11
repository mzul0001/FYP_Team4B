import keras
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array,load_img

##from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras.layers import Flatten,Dense, Dropout,Activation
from keras.optimizers import Adam
##from keras.callbacks import EarlyStopping

##from keras.optimizers import RMSprop
from keras.models import model_from_json
import os

from PIL import Image




def PreprocessData(x_train,y_train):
    #feature scalling (normalization) on image data
    
    ##min 0 max 1, RGB is 0 to 255, thus it devides by 255
    x_train = x_train.astype('float32') /255
    ##x_test = x_test.astype('float32') /255
    
    ##mnist data type is (batch size (or sample),width,height)
    ##keras data type is (batch size,width,height,channel)
    
    x_train = x_train.reshape((int(x_train.shape[0]),28,28,1)) ##28**28 = 784, 26658/784
    ##x_test = x_test.reshape((213,28,28,1))
    

    ##label is One-hot encoded ---all avlues are 1 or 0 which avoids errors in learning
    y_train = keras.utils.to_categorical(y_train)  
    ##y_test = keras.utils.to_categorical(y_test)

    return x_train,y_train





def CreateModel():
    
    #construct model
    ##conv2D(filter--number of outputs,kernel_size (n*n) padding same--output size is same with original input size,
    ##input data type)
    model = Sequential()

    #first layer
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1', input_shape=(28,28,1)) )
    model.add(MaxPooling2D(pool_size=(2,2)) )
    ##model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    #Second, third layer
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2', input_shape=(28,28,1)) )
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3', input_shape=(28,28,1)) )
    
    model.add(MaxPooling2D(pool_size=(2,2)) )
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    #4th,5th,6th layer
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4', input_shape=(28,28,1)) )
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv5', input_shape=(28,28,1)) )
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv6', input_shape=(28,28,1)) )
    model.add(MaxPooling2D(pool_size=(2,2)) )
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    #7th,8th,9th layer
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv7', input_shape=(28,28,1)) )
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv8', input_shape=(28,28,1)) )
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv9', input_shape=(28,28,1)) )
    ##model.add(MaxPooling2D(pool_size=(2,2)) )
    model.add(Dropout(0.3))
    ##model.add(BatchNormalization())

    
    
    model.summary()

    ##prediction layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, activation='softmax'))##number of emotions
    return model



def SaveModel(model):
    model_json = model.to_json()
    with open('model.json','w') as json_file:
        json_file.write(model_json)

    model.save_weights('model.h5')
    print("Saved model")

def LoadData(x_temp,y_temp,fileName,label):
    #count = 0
    ##Imgs = glob.glob('./JAFFE/'+fileName+'/*.'+'tiff')
    Imgs = glob.glob(fileName+'/*.'+'jpg')
    for img in Imgs:
        ##if count> 5000: break
        temp = img_to_array( load_img(img,target_size = (28,28),color_mode='grayscale') )
        x_temp.append(temp)
        y_temp.append(label)
        #count +=1

def PrepareData(x_train,y_train,address):
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

##    address = './img_recognition/validation/'+fileName
##    LoadData(x_test,y_test,address +'angry',0)
##    LoadData(x_test,y_test,address +'disgust',1)
##    LoadData(x_test,y_test,address +'fear',2)
##    LoadData(x_test,y_test,address +'happy',3)
##    LoadData(x_test,y_test,address +'sad',4)
##    LoadData(x_test,y_test,address +'surprise',5)
##    LoadData(x_test,y_test,address +'neutral',6)
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
##    x_test = np.array(x_test)
##    y_test = np.array(y_test)
    #check data

    size = int(x_train.shape[0]) /(28*28)
    ##print(size)
    print('X_train:', x_train.shape, 'y_train:', y_train.shape)
##    print('X_test:', x_test.shape, 'y_test:', y_test.shape)

    #display data 
##    plt.figure(figsize=(20,20))
##    for i in range(50):
##        plt.subplot(8,8,i+1)
##        plt.imshow(x_train[i].astype('uint8'))
##        plt.axis("off")
##        plt.title(str(y_train[i]),fontsize=14)
##        
##    plt.tight_layout()
##    plt.show()
    #---------finished loading dataset-----------------------------------------

    #preprocess dataset so that keras can process them
    x_train,y_train = PreprocessData(x_train,y_train)
    return x_train,y_train








def CNN_make(x_train,y_train):
    address = './img_recognition/train/'
    x_train,y_train = PrepareData(x_train,y_train,address)
    
    model = CreateModel()

    #compile
    model.compile(loss= 'categorical_crossentropy',optimizer ='adam' , metrics=['accuracy'])

    #learning
    model.fit(x_train,y_train,batch_size= 64,epochs= 30)

    #save model
    SaveModel(model)


    #evaluate
##    loss,acc =model.evaluate(x_test,y_test)
##    print(acc,loss)
    input("enter something")    
    

def CNN_predict(img,SavedModel):
##    with open('model.json','r') as json_file:
##        SavedModel = model_from_json(json_file.read())
##
##    SavedModel.load_weights('model.h5')
##
##    SavedModel.compile(loss= 'categorical_crossentropy',optimizer ='adam' , metrics=['accuracy'])

    x_test = []
    
    temp = img_to_array(img)
    ##temp = img_to_array( load_img(img,target_size = (28,28),color_mode='grayscale') )
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
    
    

##def CNN_predict_array(x_test,y_test):
##    with open('model.json','r') as json_file:
##        SavedModel = model_from_json(json_file.read())
##
##    SavedModel.load_weights('model.h5')
##
##    SavedModel.compile(loss= 'categorical_crossentropy',optimizer ='adam' , metrics=['accuracy'])
##
##    
##    ##Load only testing data(x_test)
##    Imgs = sorted(glob.glob('./outputs/*.'+'jpg'), key=os.path.getmtime)
##    sorted(Imgs, key=os.path.getctime)
##    for img in Imgs:
##        ##print(img)
##        temp = img_to_array( load_img(img,target_size = (28,28),color_mode='grayscale') )
##        x_test.append(temp)
##        
##
##
##
##    x_test = np.array(x_test)
##    ##y_test = np.array(y_test)
##
##    
##    
##    x_test = x_test.astype('float32') /255
##    ##x_test = x_test.reshape((213,28,28,1))
##    
##    ##y_test = keras.utils.to_categorical(y_test)
##    
##    #predict and obtain the array of corresponding classes
##    label = SavedModel.predict_classes(x_test)  #testing data 1 to 10 if x_test[1:10,]
##
##    array = []
##    for item in label:
##        if item ==0:
##            array.append("angry")
##        elif item == 1:
##            array.append("disgust")
##        elif item == 2:
##            array.append("fear")
##        elif item == 3:
##            array.append("happy")
##        elif item == 4:
##            array.append("sad")
##        elif item == 5:
##            array.append("surprise")
##        elif item == 6:
##            array.append("neutral")
##
##    file = open('label.txt','w')
##    for i in range(len(array)):
##        ##print(i+1,array[i])
##        file.write(str(array[i])+' '+Imgs[i] +'\n')
##    file.close()
##            
##    return label, Imgs


    

def CNN_evaluate(x_train,y_train):
    address = './img_recognition/train/'
    x_test = []
    y_test = []
    x_test,y_test = PrepareData(x_train,y_train,address)


    with open('model.json','r') as json_file:
        SavedModel = model_from_json(json_file.read())
    SavedModel.load_weights('model.h5')
    SavedModel.compile(loss= 'categorical_crossentropy',optimizer ='adam' , metrics=['accuracy'])

    
    
    score = SavedModel.evaluate(x_test, y_test, batch_size= 64,verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':

    x_train =[]
    y_train =[]
    x_test = []
    y_test = []
    
    #newly create model
    ##CNN_make(x_train,y_train)

    #load model and
    #label,Imgs = CNN_predict(x_test,y_test)

    CNN_evaluate(x_test,y_test)
    

##    
##    array = []
##    for item in label:
##        if item ==0:
##            array.append("angry")
##        elif item == 1:
##            array.append("disgust")
##        elif item == 2:
##            array.append("fear")
##        elif item == 3:
##            array.append("happy")
##        elif item == 4:
##            array.append("sad")
##        elif item == 5:
##            array.append("surprise")
##        elif item == 6:
##            array.append("neutral")
##
##    plt.figure(figsize=(20,20))
##    for i in range(20):
##        plt.subplot(5,5,i+1)
##        plt.imshow(array_to_img(x_test[i]))
##        plt.axis("off")
##        plt.title(str(Imgs[i]),fontsize=12)
##    plt.tight_layout()
##    plt.show()
##
##
##    file = open('label.txt','w')
##    for i in range(len(array)):
##        ##print(i+1,array[i])
##        file.write(str(array[i])+' '+Imgs[i] +'\n')
##    file.close()


    

