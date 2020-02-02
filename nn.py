from tensorflow.keras.models import model_from_json
from tensorflow.keras import optimizers
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def predict(features, model):
    '''
    function predicts the category of the input using a neural network model
    precondition:
    :param features: the model input
    postcondition:
    :return: the predicted category of the input
    '''
    # predict the class
    label = model.predict_classes(features)
    if label == 0:
        return 'angry'
    elif label == 1:
        return 'disgust'
    elif label == 2:
        return 'fear'
    elif label == 3:
        return 'happy'
    elif label == 4:
        return 'neutral'
    elif label == 5:
        return 'sad'
    elif label == 6:
        return 'surprise'


def loadModel(model_file, weight_file, lr=0.001):
    '''
    function load the neural network model for the emotion classification process
    precondition:
    :param:
    postcondition:
    :return: the loaded neural network model
    '''
    print('Loading model...')
    # load json and create model
    with open(model_file, 'r') as json_file:
        loaded_model = model_from_json(json_file.read())
    # load weights into new model
    loaded_model.load_weights(weight_file)

    # Adam (Adaptive Moment Estimation) -- an algorithm for first-order gradient-based optimization of stochastic
    #                                      objective functions, based on adaptive estimates of lower-order moments
    #                                   -- computationally efficient, has little memory requirements
    #                                   -- suited for problems that are large in terms of data and/or parameters
    # learning_rate -- Adam's default learning rate is 0.001
    #               -- reduced to avoid overfitting while attempting to increase accuracy
    optimizer = optimizers.Adam(learning_rate=lr)

    # categorical cross entropy -- also called multi class log loss
    #                           -- measures the performance of a classification
    #                              model where the prediction input is a probability value between 0 and 1
    # sparse -- the classes are labelled as consecutive numbers starting from 0
    loaded_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return loaded_model


def saveModel(model, model_file, weight_file):
    '''
    function saves model for distribution
    precondition:
    :param  model: the neural network model
            model_file: filename to save model in
            weight_file: filename to save weights in
    postcondition:
    :return:
    '''
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file, 'w') as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(weight_file)
    return


def trainNN(model, train_data, test_data, train_labels, test_labels, batch_size, epochs):
    '''
    function trains a neural network model
    precondition:
    :param  model: the neural network model
            train_data: the model input for training
            test_data: the model input for testing trained model
            train_labels: the model supposed output of each training input
            test_labels: the model supposed output of each testing input
    postcondition:
    :return:
    '''
    # train the neural network model  # batch_size -- the number of data fed into the model at each
    # iteration  # epochs -- the number of repetition to feed all data into the model  # verbose -- training overview
    # display method  # validation_data -- the testing data set  # the number of iterations per epoch = total
    # data/batch_size  # the total number of iterations = the number of iterations per epoch * epoch
    trained = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, verbose=2,
                        validation_data=(test_data, test_labels), shuffle=True)
    print("Evaluating NN...")
    [loss, accuracy] = model.evaluate(test_data, test_labels)
    print("Loss = {}, accuracy = {}".format(loss, accuracy))
    predicted = model.predict_classes(test_data)
    category = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']
    print(classification_report(test_labels, predicted, target_names=category))

    # # check for overfitting/underfitting
    # # loss curve
    # plt.plot(trained.history['loss'], 'r')
    # plt.plot(trained.history['val_loss'], 'b')
    # plt.legend(['train', 'test'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve')
    # plt.show()
    #
    # # accuracy curve
    # plt.plot(trained.history['accuracy'], 'r')
    # plt.plot(trained.history['val_accuracy'], 'b')
    # plt.legend(['train', 'test'])
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy Curve')
    # plt.show()
    return model
