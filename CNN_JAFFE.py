import keras
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array,load_img

##from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Flatten,Dense, Dropout,Activation
from keras.optimizers import Adam
##from keras.callbacks import EarlyStopping

##from keras.optimizers import RMSprop






def PreprocessData(x_train,y_train,x_test,y_test):
    #feature scalling (normalization) on image data
    
    ##min 0 max 1, RGB is 0 to 255, thus it devides by 255
    x_train = x_train.astype('float32') /255
    x_test = x_test.astype('float32') /255
    
    ##mnist data type is (batch size (or sample),width,height)
    ##keras data type is (batch size,width,height,channel)
    x_train = x_train.reshape((213,28,28,1)) ##28**28 = 784, 2617344/784=3338
    x_test = x_test.reshape((213,28,28,1))
    

    ##label is One-hot encoded ---all avlues are 1 or 0 which avoids errors in learning
    y_train = keras.utils.to_categorical(y_train)  
    y_test = keras.utils.to_categorical(y_test)

    return x_train,y_train,x_test,y_test





def CreateModel():
    
    #construct model
    ##conv2D(filter--number of outputs,kernel_size (n*n) padding same--output size is same with original input size,
    ##input data type)
    model = Sequential()

    #first layer
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1', input_shape=(28,28,1)) )
    model.add(MaxPooling2D(pool_size=(2,2)) )
    ##model.add(Activation('relu'))
    ##model.add(Dropout(0.3))

    #Second layer
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2', input_shape=(28,28,1)) )
    model.add(MaxPooling2D(pool_size=(2,2)) )

    #Third layer
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3', input_shape=(28,28,1)) )
    ##model.add(MaxPooling2D(pool_size=(2,2)) )
    
    model.summary()

    ##prediction layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, activation='softmax'))##number of emotions
    return model


def LoadData(x_temp,y_temp,fileName,label):
    Imgs = glob.glob('./JAFFE/'+fileName+'/*.'+'tiff')
    for img in Imgs:
        temp = img_to_array( load_img(img,target_size = (28,28),color_mode='grayscale') )
        x_temp.append(temp)
        y_temp.append(label)

def CNN():

    #------load data x is image, y is label (ex 1,5,3...)  -------------------
    ##(x_train, y_train), (x_test, y_test) = mnist.load_data()
    ##0 angry,
    ##1 disgust,
    ##2 fear,
    ##3 happy,
    ##4 sad,
    ##5 surprise,
    ##6 neutral    
    x_train =[]
    y_train =[]
    x_test = []
    y_test = []

    
    LoadData(x_train,y_train,'angry',0)
    LoadData(x_train,y_train,'disgust',1)
    LoadData(x_train,y_train,'fear',2)
    LoadData(x_train,y_train,'happy',3)
    LoadData(x_train,y_train,'sad',4)
    LoadData(x_train,y_train,'surprise',5)
    LoadData(x_train,y_train,'neutral',6)

    LoadData(x_test,y_test,'angry',0)
    LoadData(x_test,y_test,'disgust',1)
    LoadData(x_test,y_test,'fear',2)
    LoadData(x_test,y_test,'happy',3)
    LoadData(x_test,y_test,'sad',4)
    LoadData(x_test,y_test,'surprise',5)
    LoadData(x_test,y_test,'neutral',6)
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    #check data

    
    print('X_train:', x_train.shape, 'y_train:', y_train.shape)
    print('X_test:', x_test.shape, 'y_test:', y_test.shape)

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
    x_train,y_train,x_test,y_test = PreprocessData(x_train,y_train,x_test,y_test)

    model = CreateModel()

    #compile
    model.compile(loss= 'categorical_crossentropy',optimizer ='adam' , metrics=['accuracy'])

    #learning
    model.fit(x_train,y_train,batch_size= 64,epochs= 30)



    #predict or evaluate
    loss,acc =model.evaluate(x_test,y_test)
    print(acc)
    input("enter something")
    
    
    
    
    


    
    

    
    
    
    


    

    

if __name__ == '__main__':
    CNN()
