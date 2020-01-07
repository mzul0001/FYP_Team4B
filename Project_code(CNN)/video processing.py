import cv2
import numpy    
import moviepy.editor as mp
from CNN import CNN_predict
import keras
from keras.models import model_from_json
def ImageProcessing(img,FaceNo,model):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization
    gray = cv2.equalizeHist(gray)

    # Image Sharpening with Gaussian Blur
    gaussian = cv2.GaussianBlur(gray, (9, 9), 10.0)
    gray = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0, gray)
    
    #obtain cascade classifiers
    FaceAddress ='C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python37-32\\'
    FaceAddress +='Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
    FaceCascade = cv2.CascadeClassifier(FaceAddress)



    color = (0, 0, 255)

    ##execute face detections on gray colored image
    ##detectMultiScale(Mat image, MatOfRect objects, double scaleFactor,
    ##                 int minNeighbors, int flag, Size minSize, Size maxSize)
    ##Scalse factor --how much a given image is shrunk to be processed
    ##minNeighbours -- If the value is bigger, it detects less objects but less misdetections. If smaller, it detects more objects but more misdetections.

   
    # For each resulting detection, `levelWeights` will then contain the certainty of classification at the final stage.
    faces, _, levelWeights = FaceCascade.detectMultiScale3(image=gray, scaleFactor=1.3, minNeighbors=6,
                                                       outputRejectLevels=True)
    labels = []
    

    ##draw a rectangle for each face
    for (x,y,width,height) in faces:
        ##image, left up coordinate, right down coordinate, color, thickness
        img = cv2.rectangle(img,(x,y),(x+width,y+height),color,2)

        Face = img[y:y+height,x:x+width]
        Gray = cv2.cvtColor(Face, cv2.COLOR_BGR2GRAY) 
        resized = cv2.resize(Gray,(28,28))
        
        label=CNN_predict(resized,model)
        labels.append(label)
        img = cv2.putText(img = img, text = str(label), org =(x, y),fontFace = cv2.FONT_HERSHEY_SIMPLEX ,fontScale =1.0, color =(255, 255, 255), thickness=1)
          
            
        
    return img,labels

def VideoProcessing(videoname):
    ##read a video file
    video = cv2.VideoCapture(videoname +'.mp4')


    Video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    Video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps= video.get(cv2.CAP_PROP_FPS)##the number of frames per sec
    ##fourcc is MJPG,DIVX or XVID 
    writer = cv2.VideoWriter("Tagged_video.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (Video_width, Video_height))
    file = open('label.txt','w')

    with open('model.json','r') as json_file:
        SavedModel = model_from_json(json_file.read())
    SavedModel.load_weights('model.h5')
    SavedModel.compile(loss= 'categorical_crossentropy',optimizer ='adam' , metrics=['accuracy'])

    FaceNo =0
    
    labels = []
    
    while video.isOpened():
        #read a fram from video
        ret, frame = video.read()
     
        #if there is no next frame, the loop terminates
        if not ret: break
     
        #----------------- start Image processing----
        
       
        
        current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        timestamp = current_frame / fps

        #video cropped for only 15 seconds for testing 
        if timestamp > 8: break
        ##imgname = str(timestamp)+'_' + str(current_frame) +'_'
        
        frame,labels = ImageProcessing(frame,FaceNo,SavedModel)

        for emotion in labels:
            file.write(str(timestamp)+' '+ str(FaceNo) +' ' + str(emotion) +'\n')
            FaceNo +=1
        
        writer.write(frame)
        #-----------------


     
        #stop its execution by pressing Q-key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
     
        #release memory
    file.close()
    writer.release()
    video.release()
    cv2.destroyAllWindows()
    ##out.release()








    
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

    VideoProcessing('sample2') ##detect face,save the image and return the total number of detected face
##    x_test,y_test = []
##    label,Imgs = CNN_predict(x_test,y_test)
##    VideoTagging('')
    ##FE(imgNo)
        
    
    
