import cv2
import numpy


def IP(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #obtain cascade classifiers
    address ='C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python37-32\\'
    address +='Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(address)


    color = (0, 0, 255)
    ##execute face detections on gray colored image
    ##(image, scaleFactor,minNeighbours)
    faces = cascade.detectMultiScale(gray, 1.3, 6)
    count =0

    ##save faces in a directory
    for (x,y,width,height) in faces:
        
        
        ##cut the face
        savedimg = img[y:y+height,x:x+width]
        #name of the image
        imgname = "testcase "+ str(count) + '.jpg'
        #save the image into the outputs file
        cv2.imwrite("./outputs/" +imgname, savedimg)
        count += 1

    ##draw a rectangle for each face
    for (x,y,width,height) in faces:
        ##image, left up coordinate, right down coordinate, color, thickness
        img = cv2.rectangle(img,(x,y),(x+width,y+height),color,2)


    return img
    
#動画を読込み
#カメラ等でストリーム再生の場合は引数に0等のデバイスIDを記述する
video = cv2.VideoCapture('sample2b.mp4')
boolean = False

if video.isOpened():
    boolean = True
    # フレームを読込み
    ret, frame = video.read()

    frame = frame.transpose(0, 1, 2)  # [::-1]##rotate img to be correct orientation
    height, width, layers = frame.shape
    height, width = int(height*0.702), int(width*0.702)
    size = (width, height)
    out = cv2.VideoWriter(filename='output.mp4', apiPreference=0, fourcc=cv2.VideoWriter_fourcc(*'MP4V'), fps=15, frameSize=size)

while boolean:
    ret, frame = video.read()

    #フレームが読み込めなかった場合は終了（動画が終わると読み込めなくなる）
    if not ret: break

    #-----------------
    # 画像処理を記述する
    frame = frame.transpose(0,1,2)#[::-1]##rotate img to be correct orientation
    frame = cv2.resize(frame, dsize=None, fx=0.702, fy=0.702) # resize the img
    frame = IP(frame) # draw a rectangle
    #-----------------

    #フレームの描画
    out.write(frame)
    cv2.imshow('frame', frame)
 
    #qキーの押下で処理を中止
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

#メモリの解放
video.release()
cv2.destroyAllWindows()
out.release()
