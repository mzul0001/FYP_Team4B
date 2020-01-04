import cv2
import numpy    
import moviepy.editor as mp
def ImageProcessing(img,imgname):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #obtain cascade classifiers
    FaceAddress ='C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python37-32\\'
    FaceAddress +='Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
    FaceCascade = cv2.CascadeClassifier(FaceAddress)

##    EyeAddress ='C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python37-32\\'
##    EyeAddress +='Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml'
##    EyeCascade = cv2.CascadeClassifier(EyeAddress)
##
##    MouthAddress ='C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python37-32\\'
##    MouthAddress +='Lib\\site-packages\\cv2\\data\\haarcascade_mcs_mouth.xml'
##    MouthCascade = cv2.CascadeClassifier(MouthAddress)


    color = (0, 0, 255)
    color2 =(0,255,0)
    color3 = (0,0,0)
    ##execute face detections on gray colored image
    ##detectMultiScale(Mat image, MatOfRect objects, double scaleFactor,
    ##                 int minNeighbors, int flag, Size minSize, Size maxSize)
    ##Scalse factor --how much a given image is shrunk to be processed
    ##minNeighbours -- If the value is bigger, it detects less objects but less misdetections. If smaller, it detects more objects but more misdetections.

    faces = FaceCascade.detectMultiScale(image = gray, scaleFactor = 1.25,minNeighbors= 2)


    FaceNo =0

    ##draw a rectangle for each face
    for (x,y,width,height) in faces:
        ##image, left up coordinate, right down coordinate, color, thickness
        img = cv2.rectangle(img,(x,y),(x+width,y+height),color,2)
        saved = img[y:y+height,x:x+width]
        #name of the image
        imgname = imgname+ str(FaceNo)+'.jpg'
        #save the image into the outputs file
        cv2.imwrite("./outputs/" +imgname, saved)
        FaceNo += 1
        

##        #----------------eyes detection on a face-------------------
##        Eyes = EyeCascade.detectMultiScale(image = img,minNeighbors=30)
##        for (x,y,width,height) in Eyes:
##        ##image, left up coordinate, right down coordinate, color, thickness
##            img = cv2.rectangle(img,(x,y),(x+width,y+height),color2,2)
##        #-----------------------------------------------------------
##
##        #----------------Mouth detection on a face------------------
##        Mouths = MouthCascade.detectMultiScale(image = img,scaleFactor = 1.25,minNeighbors= 30)
##        for (x,y,width,height) in Mouths:
##        ##image, left up coordinate, right down coordinate, color, thickness
##            img = cv2.rectangle(img,(x,y),(x+width,y+height),color3,2)
##        #-----------------------------------------------------------





    ##cv2.imwrite('highschool_tested.jpg', img)
    ##cv2.imshow('img',img)

    return img

def VideoProcessing(videoname):
    ##read a video file
    video = cv2.VideoCapture(videoname +'.mp4')
    count =0 ##count is the number of images detected and saved
    boolean = False
    if video.isOpened():
        boolean = True
        ret, frame = video.read()
        frame = cv2.resize(frame, dsize=None, fx=0.702, fy=0.702) # resize the img
        height, width, layers = frame.shape
        height, width = int(height*0.702), int(width*0.702)
        size = (width, height)
        out = cv2.VideoWriter(filename='output.mp4', apiPreference=0, fourcc=cv2.VideoWriter_fourcc(*'MP4V'), fps=15, frameSize=size)


    Video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    Video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps= video.get(cv2.CAP_PROP_FPS)##the number of frames per sec
    ##fourcc is MJPG,DIVX or XVID 
    writer = cv2.VideoWriter("test_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (Video_width, Video_height))

    while video.isOpened():
        #read a fram from video
        ret, frame = video.read()
     
        #if there is no next frame, the loop terminates
        if not ret: break
     
        #----------------- start Image processing----
        
        ##frame = frame.transpose(1,0,2)[::-1]##rotate img from landscape to portrait

         
        ##frame = cv2.resize(frame, dsize=None, fx=0.7, fy=0.7) # resize the img

        ##cv2.waitKey(1)
        
        current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        timestamp = current_frame / fps
        imgname = str(timestamp)+'_' + str(current_frame) +'_'
        
        frame= ImageProcessing(frame,imgname) # detect a face in an image
        writer.write(frame)
        #-----------------

        out.write(frame)
        cv2.imshow('frame', frame)
     
        #stop its execution by pressing Q-key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
     
        #release memory
    writer.release()
    video.release()
    cv2.destroyAllWindows()
    out.release()
    

 
    
if __name__ == '__main__':

    VideoProcessing('sample3') ##detect face,save the image and return the total number of detected face
    
    ##FE(imgNo)
        
    
    
