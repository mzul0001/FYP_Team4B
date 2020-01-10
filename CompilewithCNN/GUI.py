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
    loginwindow.geometry("500x200+520+250")
    loginwindow.title("Login")

    detailslabel = tk.Label(loginwindow, text="Please fill your login credentials!!")
    detailslabel.config(font=("Times New Roman", 20))
    detailslabel.place(relx=0.15, rely=0.1)

    userlabel = tk.Label(loginwindow, text="Username:")
    userlabel.config(font=("Arial", 10))
    userlabel.place(relx=0.15, rely=0.35)

    passwordlabel = tk.Label(loginwindow, text="Password:")
    passwordlabel.config(font=("Arial", 10))
    passwordlabel.place(relx=0.15, rely=0.5)

    registerbutton = tk.Button(loginwindow, text="Register", width=20, command=registerswindow)
    registerbutton.place(relx=0.1, rely=0.70)

    global username_verification
    global password_verification

    username_verification=tk.StringVar()
    password_verification=tk.StringVar()

    global username_entry
    global password_entry

    username_entry=tk.Entry(loginwindow,textvariable=username_verification)
    username_entry.config(font=("Arial", 10))
    username_entry.place(relx=0.4,rely=0.35)

    password_entry=tk.Entry(loginwindow,textvariable=password_verification,show='*')
    password_entry.config(font=("Arial",10))
    password_entry.place(relx=0.4,rely=0.5)

    loginbutton = tk.Button(loginwindow, text="Login", width=20, command=verifylogin)
    loginbutton.place(relx=0.6, rely=0.68)

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
    loginsucess1.title("YAY")
    loginsucess1.geometry("200x150+700+300")
    loginlabel = tk.Label(loginsucess1, text="Sucessfully login", fg="green")
    loginlabel.config(font=("Arial", 10))
    loginlabel.place(relx=0.15, rely=0.1)
    loginsuccessbutton = tk.Button(loginsucess1, text="OKAY", width=10, command=destroy1)
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
        registersuccess = tk.Label(registerwindow, text="Registration sucess! Please close the screen to login ", fg="green")
        registersuccess.place(relx=0.2, rely=0.7)
        registersuccess.after(1000,registersuccess.destroy)
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
            registersuccess = tk.Label(registerwindow, text="Registration sucess! Please close the screen to login ", fg="green")
            registersuccess.place(relx=0.2, rely=0.7)
            registersuccess.after(1000, registersuccess.destroy)
        else:
            registerfail = tk.Label(registerwindow, text="Registration fail! Please use a new username", fg="red")
            registerfail.place(relx=0.2, rely=0.7)
            registerfail.after(1000,registerfail.destroy)
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
    label=tk.Label(help_window,text="Welcome to Help Page")
    label.config(font=("Arial", 30))
    label.place(relx=0.20, rely=0.05)

    help_window.mainloop()

def thirdWindow(filename):
    VideoProcessing(filename)
    global outputwindow
    outputwindow=tk.Toplevel(mainwindow)
    #outputwindow = tk.Tk()
    outputwindow.geometry("1920x1080")
    outputwindow.title("Output")
    # label = tk.Label(outputwindow, text="Output")
    # label.config(font=("Arial", 30))
    # label.place(relx=0.1, rely=0.1)

    bvideo_name = "Tagged_video.mp4"  # This is your video file path

    clip = mp.VideoFileClip(bvideo_name)
    clip_resized = clip.resize(height=360)  # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
    clip_resized.write_videofile("Tagged_video1.mp4")
    video_name="Tagged_video1.mp4"
    #os.system(video_name)
    video = imageio.get_reader(video_name)
    my_label = tk.Label(outputwindow)
    my_label.place(relx=0.0,rely=0.0)
    thread = threading.Thread(target=stream, args=(my_label,video))
    thread.daemon = 1
    thread.start()
    outputwindow.mainloop()
    # filename=filedialog.askopenfilename()
    # processedfile=open(filename,'r')
    # print(processedfile)

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

    uploadimage=tk.PhotoImage(file=r"C:\Users\YU GIN\Desktop\FYP 2\uploadbutton.gif")
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
