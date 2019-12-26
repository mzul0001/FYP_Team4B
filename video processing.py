import cv2
import numpy


import cv2

    

def ImageProcessing(img,count):
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


    

    ##draw a rectangle for each face
    for (x,y,width,height) in faces:
        ##image, left up coordinate, right down coordinate, color, thickness
        img = cv2.rectangle(img,(x,y),(x+width,y+height),color,2)
        saved = img[y:y+height,x:x+width]
        #name of the image
        imgname = "face"+ str(count) + '.jpg'
        #save the image into the outputs file
        cv2.imwrite("./outputs/" +imgname, saved)
        count += 1
        

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
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img,count

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

        

    while video.isOpened():
        #read a fram from video
        ret, frame = video.read()
     
        #if there is no next frame, the loop terminates
        if not ret: break
     
        #----------------- start Image processing----
        
        ##frame = frame.transpose(1,0,2)[::-1]##rotate img from landscape to portrait

        
        frame = cv2.resize(frame, dsize=None, fx=0.7, fy=0.7) # resize the img

        cv2.waitKey(1)
        frame,count = ImageProcessing(frame,count) # detect a face in an image
        #-----------------

        out.write(frame)
        cv2.imshow('frame', frame)
     
        #stop its execution by pressing Q-key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
     
    #release memory
    video.release()
    cv2.destroyAllWindows()
    out.release()

if __name__ == '__main__':
    VideoProcessing('sample3')
    
    
