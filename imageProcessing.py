import cv2
import numpy as np
from extractFeatures import extractFeatures
from nn import predict
from keras.preprocessing.image import img_to_array


def enhanceImage(img):
    '''
    function enhance image quality
    preconditions:
    :param img: the image to enhance
    postcondition: the original image is not modified
    :return: the enhanced image
    '''
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # histogram equalization
    gray_img = cv2.equalizeHist(gray_img)
    # sharpen image with Gaussian Blur
    gaussian = cv2.GaussianBlur(gray_img, (9, 9), 10.0)
    gray_img = cv2.addWeighted(gray_img, 1.5, gaussian, -0.5, 0, gray_img)
    return gray_img


def processImage(img, model):
    '''
    function draws rectangle around the face detected
    preconditions:
    :param img: the image to process
           model: the neural network model to run the emotion classification
    postcondition: the original image is not modified
    :return: the processed image and the emotion classified
    '''
    gray_img = enhanceImage(img)

    # obtain cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    color = (0, 0, 255)

    # detect face(s)
    # scalse factor -- how much a given image is shrunk to be processed
    # minNeighbours -- if the value is bigger, it detects less objects but less misdetections
    #               -- if smaller, it detects more objects but more misdetections.
    # level_weights -- contain the certainty of classification at the final stage for each resulting detection
    faces, _, level_weights = face_cascade.detectMultiScale3(image=gray_img, scaleFactor=1.3, minNeighbors=6,
                                                             outputRejectLevels=True)
    labels = []

    # draw a rectangle for each face on the original image
    for (x, y, width, height) in faces:
        # img -- image
        # (x, y) -- left up coordinate
        # (x + width, y + height) -- right down coordinate
        # color -- color
        # thickness -- thickness
        img = cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
        cropped_face = gray_img[y:y + height, x:x + width]
        cropped_face = cv2.resize(cropped_face, (48, 48))

        # predict the emotion the face displayed
        # cnn
        cropped_face = [img_to_array(cropped_face)]
        cropped_face = np.array(cropped_face)
        cropped_face = cropped_face.astype('float32')/255
        label = predict(cropped_face, model)
        # mlp
        # label = predict(extractFeatures(cropped_face), 'mlp_model.json', 'mlp_model.h5')
        labels.append(label)

        # annotate the image with the emotion the face displayed
        img = cv2.putText(img=img, text=str(label), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                          color=(255, 255, 255), thickness=1)
    return img, labels