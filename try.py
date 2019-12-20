import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

def buildNN(train_feature, test_feature, train_labels, test_labels):
    units = 10
    classes = 7
    data = train_feature.shape[1]

    model = Sequential([
        Dense(10, input_shape = (data,), activation = 'relu'),  # 1st hidden layer
        Dense(10, input_shape = (data,), activation = 'relu'),  # 2nd  hidden layer
        Dense(10, input_shape = (data,), activation = 'relu'),  # 3rd hidden layer
        Dense(units, input_shape = (data,), activation = 'relu'),   # 4th hidden layer
        Dense(units, input_shape = (data,), activation = 'relu'),   # 5th hidden layer
        Dense(units, input_shape = (data,), activation = 'relu'),   # 6th hidden layer
        Dense(units, input_shape = (data,), activation = 'relu'),   # 7th hidden layer
        Dense(units, input_shape = (data,), activation = 'relu'),   # 8th hidden layer
        Dense(units, input_shape = (data,), activation = 'relu'),   # 9th hidden layer
        Dense(units, input_shape = (data,), activation = 'relu'),   # 10th hidden layer
        Dense(classes, input_shape = (data,), activation = 'softmax')   # output layer
    ])

    '''
    configure model
    
    Adam (Adaptive Moment Estimation): 
    - an algorithm for first-order gradient-based optimization of stochastic objective functions,
    based on adaptive estimates of lower-order moments
    - computationally efficient, has little memory requirements
    - suited for problems that are large in terms of data and/or parameters
    
    categorical cross entropy:
    - also called multi class log loss 
    - measures the performance of a classification model where the prediction input is a probability value between 0 and 1
    '''
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # batch_size and epochs can be experimented to change accuracy
    trained = model.fit(train_feature, train_labels, batch_size=256, epochs=20, verbose=1, validation_data=(test_feature, test_labels))
    [loss, accuracy] = model.evaluate(test_feature, test_labels)


