import cv2
from nn import loadModel
from imageProcessing import processImage


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
    writer = cv2.VideoWriter('Tagged_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (video_width, video_height))
    # prepare the file to write the list of timestamps
    file = open('label.txt', 'w')
    model = loadModel('model.json', 'model.h5')
    face_id = 0

    while video.isOpened():
        # read a frame from the video
        ret, frame = video.read()

        # if there is no next frame, the loop terminates
        if not ret: break

        # obtain the current frame
        current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        # calculate timestamp
        timestamp = current_frame/fps
        minute = timestamp//60
        second = round(timestamp%60, 2)

        # video cropped for 10 seconds for testing
        if timestamp >= 10: break

        frame, labels = processImage(frame, model)
        # write the timestamp and the classified emotions of the processed frame to a txt file
        for emotion in labels:
            file.write(str(minute) + ' minute' + ' ' + str(second) + ' second' + ' ' + str(face_id) + ' ' + str(
                emotion) + '\n')
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
   processVideo('videoplayback.mp4')
