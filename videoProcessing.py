import cv2
from CNN import CNN_predict
# from mlp import predict
from extractFeatures import extractFeatures
from tensorflow.keras.models import model_from_json


def processImage(img, model):
    '''
    function enhance image for face detection
    preconditions:
    :param img: the image to process
           model: the neural network model to run the emotion classification
    postcondition: the original image is not modified
    :return: the processed image and the emotion classified
    '''
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # histogram equalization
    gray_img = cv2.equalizeHist(gray_img)
    # sharpen image with Gaussian Blur
    gaussian = cv2.GaussianBlur(gray_img, (9, 9), 10.0)
    gray_img = cv2.addWeighted(gray_img, 1.5, gaussian, -0.5, 0, gray_img)

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
        label = CNN_predict(cropped_face, model)
        # mlp runtime test
        # label = predict(extractFeatures(cropped_face))
        labels.append(label)

        # annotate the image with the emotion the face displayed
        img = cv2.putText(img=img, text=str(label), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                          color=(255, 255, 255), thickness=1)
    return img, labels

def loadModel():
    '''
    function load the neural network model for the emotion classification process
    precondition:
    :param:
    postcondition:
    :return: the loaded neural network model
    '''
    with open('model.json', 'r') as json_file:
        loaded_model = model_from_json(json_file.read())
    loaded_model.load_weights('model.h5')
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


def processVideo(videoname):
    '''
    function process each frame of a video
    precondition:
    :param videoname: the video file to be processed
    postcondition: the original video file is not modified
    :return:
    '''
    # read a video file
    video = cv2.VideoCapture(videoname)

    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # the number of frames per sec
    fps = video.get(cv2.CAP_PROP_FPS)

    # prepare the file to write the combined processed frames
    writer = cv2.VideoWriter("Tagged_video.mp4", cv2.VideoWriter_fourcc(*"MP4V"), fps, (video_width, video_height))
    # prepare the file to write the list of timestamps
    file = open('label.txt', 'w')

    # load the neural network model
    loaded_model = loadModel()
    face_id = 0

    while video.isOpened():
        # read a frame from the video
        ret, frame = video.read()

        # if there is no next frame, the loop terminates
        if not ret: break

        # obtain the current frame
        current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        # calculate timestamp
        timestamp = current_frame / fps

        # # video cropped for only 3 seconds for testing
        # if timestamp >= 3: break

        frame, labels = processImage(frame, loaded_model)
        # write the timestamp and the classified emotions of the processed frame to a txt file
        for emotion in labels:
            file.write(str(timestamp) + ' ' + str(face_id) + ' ' + str(emotion) + '\n')
            face_id += 1
        # combine the processed frame to an mp4 file
        writer.write(frame)

        # stop its execution by pressing Q-key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
    # release memory
    file.close()
    writer.release()
    video.release()
    cv2.destroyAllWindows()
    return


##def VideoTagging(videoname):
##    #read label text file
##    file = open('label.txt')
##    text=file.read()
##    file.close()
##    lines = text.split('\n')
##    label =[]
##    for line in lines:
##        #emotion AND ./outputs\timestamp...
##        temp = line.split(' ')
##        label.append(temp[0])
##        
##    
##    video = cv2.VideoCapture(videoname +'.avi')
##    
##    Video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
##    Video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
##    fps= video.get(cv2.CAP_PROP_FPS)##the number of frames per sec
##    ##fourcc is MJPG,DIVX or XVID 
##    writer = cv2.VideoWriter("tagged_video.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (Video_width, Video_height))
##
##    while video.isOpened():
##        
##        
##        
##        #read a fram from video
##        ret, frame = video.read()
##     
##        #if there is no next frame, the loop terminates
##        if not ret: break
##        
##        current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
##        timestamp = current_frame / fps
##        #video cropped for only 15 seconds for testing 
##        if timestamp > 15: break
##
##        frame= ImageProcessing(frame,'',1,label) # detect a face in an image
##        writer.write(frame)
##
##    writer.release()
##    video.release()
##    cv2.destroyAllWindows()
##        
## 

if __name__ == '__main__':
   processVideo('sample2a.mp4')  ##detect face,save the image and return the total number of detected face
##    x_test,y_test = []
##    label,Imgs = CNN_predict(x_test,y_test)
##    VideoTagging('')
##FE(imgNo)
