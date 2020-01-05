import cv2
import numpy


# import moviepy.editor as mp


def ImageProcessing(img, imgName, faceNo):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Histogram Equalization
    gray = cv2.equalizeHist(gray)

    # Image Sharpening with Gaussian Blur
    gaussian = cv2.GaussianBlur(gray, (9, 9), 10.0)
    gray = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0, gray)

    # obtain cascade classifiers
    # FaceAddress ='C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python37-32\\'
    # FaceAddress +='Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
    # FaceCascade = cv2.CascadeClassifier(FaceAddress)

    # EyeAddress ='C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python37-32\\'
    # EyeAddress +='Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml'
    # EyeCascade = cv2.CascadeClassifier(EyeAddress)
    #
    # MouthAddress ='C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python37-32\\'
    # MouthAddress +='Lib\\site-packages\\cv2\\data\\haarcascade_mcs_mouth.xml'
    # MouthCascade = cv2.CascadeClassifier(MouthAddress)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    color = (0, 0, 255)
    # color2 = (0, 255, 0)
    # color3 = (0, 0, 0)

    # scale factor -- how much the image size is reduced at each image scale
    # minNeighbours -- how many neighbors each candidate rectangle should have to retain it
    #               -- If the value is bigger, it detects less objects but less misdetections. If smaller,
    #               it detects more objects but more misdetections.
    # For each resulting detection, `levelWeights` will then contain the certainty of classification at the final stage.
    faces, _, levelWeights = cascade.detectMultiScale3(image=gray, scaleFactor=1.3, minNeighbors=6,
                                                                  outputRejectLevels=True)

    # draw a rectangle for each face
    for (x, y, width, height) in faces:
        # image, left up coordinate, right down coordinate, color, thickness
        img = cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
        # crop image
        cropped = img[y:y + height, x:x + width]
        # name of the image
        imgName += str(faceNo) + '.jpg'
        # save the image into the outputs file
        cv2.imwrite("./outputs/" + imgName, cropped)
        faceNo += 1

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

    return img, faceNo


def VideoProcessing(videoName):
    # read a video file
    video = cv2.VideoCapture(videoName)
    # count is the number of images detected and saved
    count = 0
    # number of frames per second
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(filename='test_output.avi', apiPreference=0, fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                             fps=fps, frameSize=(width, height))

    while video.isOpened():
        # read a frame from the video
        ret, frame = video.read()

        # if there is no next frame, the loop terminates
        if not ret: break

        # ----------------- start Image processing----

        ##frame = frame.transpose(1,0,2)[::-1]##rotate img from landscape to portrait

        ##frame = frame.transpose(0, 1, 2)  #[::-1]##rotate img to be correct orientation
        ##frame = cv2.resize(frame, dsize=None, fx=0.7, fy=0.7)  # resize the img

        current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        timestamp = current_frame / fps
        imgName = str(timestamp) + '-' + str(current_frame) + '-'
        # detect a face in an image
        frame, count = ImageProcessing(frame, imgName, count)

        writer.write(frame)
        # cv2.imshow('frame', frame)

        # stop its execution by pressing Q-key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

    # release memory
    video.release()
    writer.release()
    cv2.destroyAllWindows()
    return count


if __name__ == '__main__':
    # detect face, save the image and return the total number of detected face
    faceNo = VideoProcessing('sample2a.mp4')
