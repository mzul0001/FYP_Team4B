import threading
import tkinter as tk
from tkinter import filedialog
import imageio
from videoprocessing import VideoProcessing
from PIL import Image,ImageTk
import os
import datetime
import moviepy.editor as mp



def browsebutton():
    filename=tk.filedialog.askopenfilename(filetypes=(("Video files", "*.mp4;*.flv;*.avi;*.mkv"),
                                       ("All files", "*.*") ))
    #need to check for output video
    updatefile=os.path.split(filename)
    filename=updatefile[1]
    print(filename)
    if str(filename).endswith('.mp4') or str(filename).endswith('.avi'):
        thirdWindow(filename)
    else:
        wrongwindow()

def loginwindow():
    global loginwindow
    loginwindow = tk.Tk()
    loginwindow.geometry("600x300+520+250")
    loginwindow.title("Login Page")


    detailslabel = tk.Label(loginwindow, text="Welcome to Team4B Emotion Detection Software")
    detailslabel.config(font=("Times New Roman", 20))
    detailslabel.place(relx=0.05, rely=0.05)

    details2label=tk.Label(loginwindow,text="**Please login if you have created a username and password",fg="red")
    details2label.config(font=("Times New Roman", 12))
    details2label.place(relx=0.05, rely=0.20)

    details3label = tk.Label(loginwindow,text="**Please register if you have not register yet ",fg="red")
    details3label.config(font=("Times New Roman", 12))
    details3label.place(relx=0.05, rely=0.30)

    details4label = tk.Label(loginwindow,text="**If you forget your credentials, re-register",fg="red")
    details4label.config(font=("Times New Roman", 12))
    details4label.place(relx=0.05, rely=0.40)

    userlabel = tk.Label(loginwindow, text="Username:")
    userlabel.config(font=("Arial", 10))
    userlabel.place(relx=0.05, rely=0.55)

    passwordlabel = tk.Label(loginwindow, text="Password:")
    passwordlabel.config(font=("Arial", 10))
    passwordlabel.place(relx=0.05, rely=0.65)

    registerbutton = tk.Button(loginwindow, text="Register", width=20, command=registerswindow)
    registerbutton.place(relx=0.05, rely=0.80)

    global username_verification
    global password_verification

    username_verification=tk.StringVar()
    password_verification=tk.StringVar()

    global username_entry
    global password_entry

    username_entry=tk.Entry(loginwindow,textvariable=username_verification)
    username_entry.config(font=("Arial", 10))
    username_entry.place(relx=0.2,rely=0.55)

    password_entry=tk.Entry(loginwindow,textvariable=password_verification,show='*')
    password_entry.config(font=("Arial",10))
    password_entry.place(relx=0.2,rely=0.65)

    loginbutton = tk.Button(loginwindow, text="Login", width=20, command=verifylogin)
    loginbutton.place(relx=0.7, rely=0.80)

    loginwindow.mainloop()

def verifylogin():
    loginusername=username_verification.get()
    loginpassword=password_verification.get()
    password_entry.delete(0, tk.END)
    username_entry.delete(0, tk.END)
    registermember = "register.txt"
    file = open(registermember, 'r')
    filecontent = file.read().splitlines()
    print(filecontent)
    flag=False
    for i in range(len(filecontent)):
        if filecontent[i]=="username "+ str(loginusername):
            if filecontent[i+1]== "password "+str(loginpassword):
                flag=True
                break
        else:
            flag=False

    if flag:
        loginsuccess()
    else:
        loginfail()

def destroy1():
    loginwindow.destroy()
    firstwindow()
def loginsuccess():
    global loginsucess1
    loginsucess1 = tk.Toplevel(loginwindow)
    loginsucess1.title("Successful")
    loginsucess1.geometry("200x150+700+300")
    loginlabel = tk.Label(loginsucess1, text="Login successfully", fg="green")
    loginlabel.config(font=("Arial", 10))
    loginlabel.place(relx=0.15, rely=0.1)
    loginsuccessbutton = tk.Button(loginsucess1, text="Proceed", width=10, command=destroy1)
    loginsuccessbutton.place(relx=0.5, rely=0.6)



def loginfail():
    global loginfail1
    loginfail1=tk.Toplevel(loginwindow)
    loginfail1.title("FAIL")
    loginfail1.geometry("200x150+700+300")
    loginlabel = tk.Label(loginfail1, text="Login Fail", fg="red")
    loginlabel.config(font=("Arial", 10))
    loginlabel.place(relx=0.15, rely=0.1)
    loginfailbutton = tk.Button(loginfail1, text="OKAY", width=10, command=loginfail1.destroy)
    loginfailbutton.place(relx=0.5, rely=0.6)

def registerswindow():
    global username_enter
    global username
    global password_enter
    global password
    global registerwindow
    registerwindow=tk.Toplevel(loginwindow)
    username = tk.StringVar()
    password = tk.StringVar()

    #registerwindow=tk.Tk()
    registerwindow.geometry("300x200+550+250")
    registerwindow.title("Register")

    registerlabel=tk.Label(registerwindow,text="Please fill your details")
    registerlabel.config(font=("Arial", 20))
    registerlabel.place(relx=0.10, rely=0.03)

    usernamelabel=tk.Label(registerwindow,text="Username: ")
    usernamelabel.config(font=("Arial", 10))
    usernamelabel.place(relx=0.1, rely=0.25)

    passwordlabel = tk.Label(registerwindow, text="Password: ")
    passwordlabel.config(font=("Arial", 10))
    passwordlabel.place(relx=0.1, rely=0.35)

    username_enter=tk.Entry(registerwindow,textvariable=username)
    username_enter.place(relx=0.35,rely=0.25)
    password_enter=tk.Entry(registerwindow,textvariable=password,show='*')
    password_enter.place(relx=0.35,rely=0.35)

    registerbutton=tk.Button(registerwindow,text="Register",width=20,command=registered_user)
    registerbutton.place(relx=0.2,rely=0.5)


def registered_user():
    username_info = username.get()
    password_info = password.get()
    registermember = "register.txt"
    file=open(registermember,'r')
    filecontent=file.read().splitlines()
    flag=False
    if len(filecontent)==0:
        file.close()
        file = open(registermember, 'w+')
        file.write("username " + str(username_info) + '\n')
        file.write("password " + str(password_info) + '\n')
        file.close()
        registersuccess = tk.Label(registerwindow, text="Registration success! Please close the screen to login ", fg="green")
        registersuccess.place(relx=0.05, rely=0.7)
        registersuccess.after(5000,registersuccess.destroy)
    else:
        for i in filecontent:
            if str(i)!="username "+str(username_info):
                flag=True
            else:
                flag=False
                break
        if flag:
            file.close()
            file = open(registermember, 'a+')
            file.write("username " + str(username_info) + '\n')
            file.write("password " + str(password_info) + '\n')
            file.close()
            registersuccess = tk.Label(registerwindow, text="Registration success! Please close the screen to login! ", fg="green")
            registersuccess.place(relx=0.05, rely=0.7)
            registersuccess.after(5000, registersuccess.destroy)
        else:
            registerfail = tk.Label(registerwindow, text="Registration fail! Please use a new username!", fg="red")
            registerfail.place(relx=0.05, rely=0.7)
            registerfail.after(5000,registerfail.destroy)
        password_enter.delete(0, tk.END)
        username_enter.delete(0, tk.END)


def wrongwindow():
    wrongwindow = tk.Tk()
    wrongwindow.geometry("300x200+650+300")
    wrongwindow.title("Error")
    label = tk.Label(wrongwindow, text="Please input a mp4 file")
    label.config(font=("Arial", 10))
    label.place(relx=0.30, rely=0.4)
    wrongwindow.mainloop()

def secondWindow():
    help_window=tk.Tk()
    help_window.geometry("600x700+450+50")
    help_window.title("Help")
    label=tk.Label(help_window,text="Help Page")
    label.config(font=("Arial", 30))
    label.place(relx=0.20, rely=0.05)

    help_window.mainloop()

def thirdWindow(filename):
    VideoProcessing(filename)
    global outputwindow
    outputwindow=tk.Toplevel(mainwindow)
    #outputwindow = tk.Tk()
    outputwindow.geometry("600x800+0+0")
    outputwindow.title("Output")
    label = tk.Label(outputwindow, text="Current Output of the Video")
    label.config(font=("Arial", 20))
    label.place(relx=0.05, rely=0.05)
    label1 =tk.Label(outputwindow,text="These are the classifications of emotions in the video")
    label.config(font=("Arial", 12))
    label1.place(relx=0.05,rely=0.09)
    playbutton = tk.Button(outputwindow,text="Play Video",command=outputvideoplay)
    playbutton.place(relx=0.05,rely=0.95)
    closebutton = tk.Button(outputwindow, text="Close", command=outputwindow.destroy)
    closebutton.place(relx=0.8, rely=0.95)
    # anger, fear, disgust, happiness, sadness, surprise
    # and contempt
    angerlabel = tk.Label(outputwindow, text="Angry Emotions Timestamp")
    label.config(font=("Arial", 12))
    angerlabel.place(relx=0.05, rely=0.12)

    fearlabel = tk.Label(outputwindow, text="Fear Emotions Timestamp")
    label.config(font=("Arial", 12))
    fearlabel.place(relx=0.05, rely=0.23)

    disgustlabel = tk.Label(outputwindow, text="Disgust Emotions Timestamp")
    label.config(font=("Arial", 12))
    disgustlabel.place(relx=0.05, rely=0.34)

    happinesslabel= tk.Label(outputwindow, text="Happy Emotions Timestamp")
    label.config(font=("Arial", 12))
    happinesslabel.place(relx=0.05, rely=0.45)

    sadnesslabel = tk.Label(outputwindow, text="Sad Emotions Timestamp")
    label.config(font=("Arial", 12))
    sadnesslabel.place(relx=0.05, rely=0.56)

    surpriselabel = tk.Label(outputwindow, text="Surprise Emotions Timestamp")
    label.config(font=("Arial", 12))
    surpriselabel.place(relx=0.05, rely=0.67)

    neutrallabel = tk.Label(outputwindow, text="Neutral Emotions Timestamp")
    label.config(font=("Arial", 12))
    neutrallabel.place(relx=0.05, rely=0.78)

    registermember = "label.txt"
    file = open(registermember, 'r')
    filecontent = file.read().split()
    print(filecontent)

    angertext=tk.Text(outputwindow,height=4,width=40)
    angertext.place(relx=0.05,rely=0.145)

    feartext = tk.Text(outputwindow, height=4, width=40)
    feartext.place(relx=0.05, rely=0.255)

    disgusttext=tk.Text(outputwindow,height=4,width=40)
    disgusttext.place(relx=0.05,rely=0.365)

    happinesstext = tk.Text(outputwindow, height=4, width=40)
    happinesstext.place(relx=0.05, rely=0.475)

    sadnesstext= tk.Text(outputwindow, height=4, width=40)
    sadnesstext.place(relx=0.05, rely=0.585)

    surprisetext = tk.Text(outputwindow, height=4, width=40)
    surprisetext.place(relx=0.05, rely=0.695)

    neutraltext = tk.Text(outputwindow, height=4, width=40)
    neutraltext.place(relx=0.05, rely=0.805)

    for i in range(2,len(filecontent),3):
        if str(filecontent[i])=="angry":
            sadnesstext.insert(tk.END,filecontent[i-2]+'\n')
        elif str(filecontent[i])=="disgust":
            disgusttext.insert(tk.END,filecontent[i-2]+'\n')
        elif str(filecontent[i])=="fear":
            feartext.insert(tk.END,filecontent[i-2]+'\n')
        elif str(filecontent[i])=="happy":
            happinesstext.insert(tk.END,filecontent[i-2] + '\n')
        elif str(filecontent[i])=="sad":
            sadnesstext.insert(tk.END,filecontent[i-2] + '\n')
        elif str(filecontent[i])=="surprise":
            surprisetext.insert(tk.END,filecontent[i-2] + '\n')
        elif str(filecontent[i])=="neutral":
            neutraltext.insert(tk.END,filecontent[i-2] + '\n')

    angertext.config(state=tk.DISABLED)
    sadnesstext.config(state=tk.DISABLED)
    disgusttext.config(state=tk.DISABLED)
    feartext.config(state=tk.DISABLED)
    surprisetext.config(state=tk.DISABLED)
    neutraltext.config(state=tk.DISABLED)
    happinesstext.config(state=tk.DISABLED)
        # outputtext.insert(tk.END,filecontent[i]+'\n')
    # video = imageio.get_reader(video_name)
    # my_label = tk.Label(outputwindow)
    # my_label.place(relx=0.0,rely=0.0)
    # thread = threading.Thread(target=stream, args=(my_label,video))
    # thread.daemon = 1
    # thread.start()
    outputwindow.mainloop()
    # filename=filedialog.askopenfilename()
    # processedfile=open(filename,'r')
    # print(processedfile)

def outputvideoplay():
    bvideo_name = "Tagged_video.mp4"  # This is your video file path

    clip = mp.VideoFileClip(bvideo_name)
    clip_resized = clip.resize(
        height=360)  # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
    clip_resized.write_videofile("Tagged_video1.mp4")
    video_name = "Tagged_video1.mp4"
    os.system(video_name)


def stream(label,video):
    for image in video.iter_data():
        frame_image = ImageTk.PhotoImage(Image.fromarray(image))
        label.config(image=frame_image)
        label.image = frame_image

def firstwindow():
    global mainwindow
    mainwindow=tk.Tk()
    mainwindow.title("Emotion Detection")
    mainwindow.geometry("600x700+450+50")


    button2=tk.Button(mainwindow,text="Help",width=20,command=secondWindow)
    button2.place(relx = 0.05, rely = 0.9)

    uploadimage=tk.PhotoImage(file="uploadbutton.gif")
    uploadimage=uploadimage.subsample(3, 3)

    uploadbutton=tk.Button(mainwindow,text="Upload",image=uploadimage,command=browsebutton)
    uploadbutton.place(relx=0.65,rely=0.65)


    label = tk.Label(mainwindow, text = "Emotion Detection")
    label.config(font=("Arial", 30))
    label.place(relx =0.25,rely=0.05)


    text1=tk.Text(mainwindow,height=20,width=40)
    text1.insert(tk.END,"Requirement for Emotion Detection \n 1. Good Quality Videos \n 2. HELLO")
    text1.config(font=("Arial",10))
    text1.place(relx=0.05,rely=0.4)

    mainwindow.mainloop()

if __name__ == '__main__':
    loginwindow()

