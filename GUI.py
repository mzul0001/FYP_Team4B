import threading
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tkinter import ttk
from videoProcessing import processVideo
from PIL import Image, ImageTk
import os


def browsebutton():
    '''
    This function is in charge of allowing the users to choose video files only which are in the format of mp4,
    flv,avi,and mkv. All the remaining file types will remain hidden
    precondition:
    :param: none
    :return: none
    '''
    filename = tk.filedialog.askopenfilename(filetypes=(("Video files", "*.mp4;*.flv;*.avi;*.mkv"),
                                                        ("All files", "*.*")))
    # need to check for output video
    # split file will split the directory and the file name
    # [1] is the file name
    updatefile = os.path.split(filename)
    filename1 = updatefile[1]
    if str(filename1).endswith('.mp4') or str(filename1).endswith('.avi') or str(filename1).endswith('.mkv') or str(
            filename1).endswith('.flv'):
        loadingwindow1(filename)
    else:
        wrongwindow()


def loadingwindow1(filename):
    '''
    This function creates a feedback window indicating processing in the backend. This make sure users know that
    there is a background processing occuring rather than closing the window if the users waits too long for the video
    to be processed
    precondition: filename must be video type file which are in the format of mp4, flv,avi,and mkv
    :param filename: This is the file we choose to process
    :return: the progress bar indicating the preprocessing of the video occuring in the back end
    '''
    mainwindow.destroy()
    global loadingwindow
    global progressbar
    loadingwindow = tk.Tk()
    loadingwindow.geometry("350x150+520+250")
    loadingwindow.title("Loading Window")
    # Creating labels
    label1 = tk.Label(loadingwindow, text="The file is currently processing. Please be patient! \n Thank you using "
                                          "our application", fg='blue')
    label1.config(font=("Arial", 11))
    label1.place(relx=0.05, rely=0.1)
    # indeterminate is that there is no displaying of time when the background is running
    progressbar = ttk.Progressbar(loadingwindow, orient=HORIZONTAL, length=200, mode='indeterminate')
    progressbar.place(relx=0.2, rely=0.45)
    #progress bar need to be started in another thread
    start_thread(filename)


def start_thread(filename):
    '''
    This function will allow the execution of the progress bar where the progress bar must run in another thread.
    This is because if the progress bar is not run in another thread, Tkinter does not supprot dual- threading , that
    will cause the whole GUI to freeze and only displaying the progress after the processing is done
    :param filename:This is the file we choose to process
    :return: the progress bar indicating the start of processing the video
    '''
    global t1
    progressbar.start()
    t1 = threading.Thread(target=processVideo, args=(filename,))
    t1.start()
    loadingwindow.after(10, check_thread)


def check_thread():
    '''
    This function keep checking whether the backend has ended or not. If the backend processes is finished, it will
    destroy the progress bar and show the output window of our software. Update_idle must also happen to update the
    progress bar
    precondition:none
    '''
    #checking whether the progress bar thread is alive or not
    if t1.is_alive():
        loadingwindow.after(10, check_thread)
        loadingwindow.update_idletasks()
    else:
        progressbar.stop()
        loadingwindow.destroy()
        thirdWindow()


def loginwindow():
    '''
    This function creates the login window where in the login window, users are able to register if they have not
    created their username and password, or users can login if they are existing users. A help button is also present
    in the login window to teach the user how to register.
    :return: GUI with all the buttons and label being created
    '''
    global loginwindow
    loginwindow = tk.Tk()
    loginwindow.geometry("600x300+520+250")
    loginwindow.title("Login Page")

    detailslabel = tk.Label(loginwindow, text="Welcome to Team4B Emotion Classification Software")
    detailslabel.config(font=("Times New Roman", 18))
    detailslabel.place(relx=0.05, rely=0.05)

    details2label = tk.Label(loginwindow, text="**Please login if you have created a username and a password", fg="red")
    details2label.config(font=("Times New Roman", 12))
    details2label.place(relx=0.05, rely=0.20)

    details3label = tk.Label(loginwindow, text="**Please register if you are not registered", fg="red")
    details3label.config(font=("Times New Roman", 12))
    details3label.place(relx=0.05, rely=0.30)

    details4label = tk.Label(loginwindow, text="**If you forget your credentials, re-register", fg="red")
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

    #converting it to string
    username_verification = tk.StringVar()
    password_verification = tk.StringVar()

    global username_entry
    global password_entry

    # allowing users to key in username and password
    username_entry = tk.Entry(loginwindow, textvariable=username_verification)
    username_entry.config(font=("Arial", 10))
    username_entry.place(relx=0.2, rely=0.55)

    password_entry = tk.Entry(loginwindow, textvariable=password_verification, show='*')
    password_entry.config(font=("Arial", 10))
    password_entry.place(relx=0.2, rely=0.65)

    loginbutton = tk.Button(loginwindow, text="Login", width=20, command=verifylogin)
    loginbutton.place(relx=0.7, rely=0.80)

    #inserting the logo in the window

    load = Image.open('4b.png')
    load = load.resize((150, 150), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    image = Label(image=render)
    image.image = render
    image.place(relx=0.70, rely=0.20)

    # creating the help button
    helpbutton = tk.Button(loginwindow, text="Help", width=20, command=helpwindow1)
    helpbutton.place(relx=0.375, rely=0.8)

    loginwindow.mainloop()


def helpwindow1():
    '''
    This function will create a window to educate the user on how to register and login.
    There are texts created step by step educating the users or user can choose to play a video where a
    step-by-step guide is shown.
    precondition:none
    :param:none
    postcondition:none
    :return:none
    '''
    loginhelpwindow = tk.Toplevel(loginwindow)
    loginhelpwindow.geometry("400x300+630+250")
    loginhelpwindow.title("Help")

    loginhelplabel = tk.Label(loginhelpwindow, text="Tips to get started!", fg='Red')
    loginhelplabel.config(font=("Arial", 20))
    loginhelplabel.place(relx=0.05, rely=0.05)

    #Text creation
    textguide = tk.Text(loginhelpwindow, height=8, width=40, fg='blue', wrap=WORD)
    textguide.place(relx=0.05, rely=0.20)
    textguide.insert(tk.END, "1.Click the register button.\n\n2.Type your desired unique username and password."
                             "\n\n3.After registration, close the register window and input the registered username "
                             "and password to login!")
    #user cannot edit the text
    textguide.config(state=tk.DISABLED)

    #the play video button
    helpplaybutton = tk.Button(loginhelpwindow, text="Play Video", width=20, command=outputvideoplay2)
    helpplaybutton.place(relx=0.05, rely=0.70)

    #the close button
    closebutton = tk.Button(loginhelpwindow, text="Close", width=20, command=loginhelpwindow.destroy)
    closebutton.place(relx=0.55, rely=0.70)


def outputvideoplay2():
    '''
    This function allows the operating system to browse through a default media player in our device. After locating
    the default media player, it will run the existing video that is in the directory.
    precondition: must be an mp4 file
    :param:none
    postcondition:none
    :return: video being played
    '''
    video_name = 'loginhelp.mp4'
    os.system(video_name)


def verifylogin():
    '''
    This function will verify the login of the user. If the user enters a username and password that is registered,
    permission is granted to use the software. However, it will indicating a login fail if the username enter a
    wrong username or password or the user has not registered yet.

    **This login will be improved by using online database where the user can store all P&C videos in their database
    or store valid information about the processed video on that database
    precondition:
    :param:none
    postcondition:none
    :return:none
    '''
    loginusername = username_verification.get()
    loginpassword = password_verification.get()
    #delete the entry text once user input
    password_entry.delete(0, tk.END)
    username_entry.delete(0, tk.END)
    registermember = "users.txt"
    #now all username and password is store in the file
    file = open(registermember, 'r')
    filecontent = file.read().splitlines()
    print(filecontent)
    flag = False
    for i in range(len(filecontent)):
        if filecontent[i] == "username " + str(loginusername):
            if filecontent[i + 1] == "password " + str(loginpassword):
                flag = True
                break
        else:
            flag = False

    if flag:
        loginsuccess()
    else:
        loginfail()


def destroy1():
    '''
    This function will destroy a window
    precondition:none
    :param:none
    postcondition: none
    :return: none
    '''
    loginwindow.destroy()
    firstwindow()


def loginsuccess():
    '''
    This function will create a window is to show the user has logged in successfully
    preconditon:password and username must be in strin
    :param: none
    postcondition:none
    :return:window indicating succesful login
    '''
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
    '''
    This function will create a window to show that the user has failed to login
    precondition:password and username must be in string
    :param:none
    postcondition:none
    :return: window indicating a failed login
    '''
    global loginfail1
    loginfail1 = tk.Toplevel(loginwindow)
    loginfail1.title("FAIL")
    loginfail1.geometry("200x150+700+300")
    loginlabel = tk.Label(loginfail1, text="Login Fail", fg="red")
    loginlabel.config(font=("Arial", 10))
    loginlabel.place(relx=0.15, rely=0.1)
    loginfailbutton = tk.Button(loginfail1, text="OKAY", width=10, command=loginfail1.destroy)
    loginfailbutton.place(relx=0.5, rely=0.6)


def registerswindow():
    '''
    This function will create a window that allows the user to register so he/she can use our software.
    precondition:none
    :param:none
    postcondition: none
    :return: none
    '''
    global username_enter
    global username
    global password_enter
    global password
    global registerwindow
    registerwindow = tk.Toplevel(loginwindow)
    username = tk.StringVar()
    password = tk.StringVar()

    # registerwindow=tk.Tk()
    registerwindow.geometry("300x200+550+250")
    registerwindow.title("Register")

    registerlabel = tk.Label(registerwindow, text="Please fill in your details!", fg='red')
    registerlabel.config(font=("Arial", 16))
    registerlabel.place(relx=0.05, rely=0.03)

    usernamelabel = tk.Label(registerwindow, text="Username: ")
    usernamelabel.config(font=("Arial", 10))
    usernamelabel.place(relx=0.05, rely=0.25)

    passwordlabel = tk.Label(registerwindow, text="Password: ")
    passwordlabel.config(font=("Arial", 10))
    passwordlabel.place(relx=0.05, rely=0.35)

    username_enter = tk.Entry(registerwindow, textvariable=username)
    username_enter.place(relx=0.35, rely=0.25)
    password_enter = tk.Entry(registerwindow, textvariable=password, show='*')
    password_enter.place(relx=0.35, rely=0.35)

    registerbutton = tk.Button(registerwindow, text="Register", width=20, command=registered_user)
    registerbutton.place(relx=0.2, rely=0.5)


def registered_user():
    '''
    This function will check whether the user has already registered or not. If the user has not registered,
    there will be no error, however if the user has registered, a pop out will show that fail registration has occured,
    and please input a new username and password
    precondition:none
    :param:none
    postcondition:none
    :return:none
    '''
    username_info = username.get()
    password_info = password.get()
    registermember = "users.txt"
    file = open(registermember, 'r')
    filecontent = file.read().splitlines()
    flag = False
    #write to empty file
    if len(filecontent) == 0:
        file.close()
        file = open(registermember, 'w+')
        file.write("username " + str(username_info) + '\n')
        file.write("password " + str(password_info) + '\n')
        file.close()
        registersuccess = tk.Label(registerwindow, text="Registration success! Please close the screen to login ",
                                   fg="green")
        registersuccess.place(relx=0.05, rely=0.7)
        #the label will be display for 5 seconds and then disappear
        registersuccess.after(5000, registersuccess.destroy)
    else:
        #username and password check
        for i in filecontent:
            if str(i) != "username " + str(username_info):
                flag = True
            else:
                flag = False
                break
        if flag:
            file.close()
            file = open(registermember, 'a+')
            file.write("username " + str(username_info) + '\n')
            file.write("password " + str(password_info) + '\n')
            file.close()
            registersuccess = tk.Label(registerwindow, text="Registration success! Please close the screen to login! ",
                                       fg="green")
            registersuccess.place(relx=0.05, rely=0.7)
            #the label will be display for 5 seconds and then disappear
            registersuccess.after(5000, registersuccess.destroy)
        else:
            registerfail = tk.Label(registerwindow, text="Registration fail! Please use a new username!", fg="red")
            registerfail.place(relx=0.05, rely=0.7)
            registerfail.after(5000, registerfail.destroy)
        password_enter.delete(0, tk.END)
        username_enter.delete(0, tk.END)


def wrongwindow():
    '''
    This function is to show the user that he/she needs to select an appropriate video file. A window will pop out
    to show that the user must input a file or a valid file must be selected in order for processing to occur.
    precondition:none
    :param:none
    postcondition:none
    :return:window showing a valid file must be selected
    '''
    wrongwindow = tk.Tk()
    wrongwindow.geometry("250x100+650+400")
    wrongwindow.title("Error")
    label = tk.Label(wrongwindow, text="Error : Please input a mp4/avi/mkv/flv file", fg='red')
    label.config(font=("Arial", 10))
    label.place(relx=0.01, rely=0.1)
    closeButton = tk.Button(wrongwindow, text="Close", width=10, command=wrongwindow.destroy)
    closeButton.place(relx=0.5, rely=0.6)
    wrongwindow.mainloop()


def secondWindow():
    '''
    This function will create a window to educate the user on how to correctly user our software so
    no errors will occur during the execution of our software. If the user is not familiar with english, a video
    will also be shown on how to use our software
    precondition:none
    :param:none
    postcondition:none
    :return:none
    '''
    help_window = tk.Tk()
    help_window.geometry("400x500+450+50")
    help_window.title("Help")
    label = tk.Label(help_window, text="Software Tutorial", fg='red')
    label.config(font=("Arial", 20))
    label.place(relx=0.05, rely=0.05)

    #Step by Step guide
    textguide = tk.Text(help_window, height=18, width=40, fg='blue', wrap=WORD)
    textguide.place(relx=0.05, rely=0.20)
    textguide.insert(tk.END, "1.After login, the main menu will pop out.\n\n2.Read through the requirements before "
                             "uploading a video.\n\n 3.Click on the Upload button and choose the video that you would "
                             "like to process.\n\n4.Kindly wait for the video to be processed.\n\n5. After processed, "
                             "the timestamp of emotion detected are shown and the video is annotated.")
    textguide.config(state=tk.DISABLED)
    closeButton = tk.Button(help_window, text="Close", width=10, command=help_window.destroy)
    closeButton.place(relx=0.6, rely=0.85)

    # user guide on using the software
    helpplaybutton = tk.Button(help_window, text="Play Video", width=20, command=outputvideoplay3)
    helpplaybutton.place(relx=0.05, rely=0.85)
    help_window.mainloop()


def outputvideoplay3():
    '''
       This function allows the operating system to browse through a default media player in our device. After locating
       the default media player, it will run the existing video that is in the directory.
       precondition: must be an mp4 file
       :param:none
       postcondition:none
       :return: video being played
       '''
    bvideo_name = "softwarehelp.mp4"  # This is your video file path
    os.system(bvideo_name)


def thirdWindow():
    '''
    This function creates the output window where in that output window will consist the timestamp of the classification
    of emotions.in the video. User can also click on the play button to see the emotions being tagged in the video itself
    precondition:a appropriate video file must be inputted
    :param:none
    postcondition:none
    :return:an output window showing time stamp of classification of emotions in the video
    '''
    global outputwindow
    # outputwindow=tk.Toplevel(mainwindow)
    outputwindow = tk.Tk()
    outputwindow.geometry("600x800+0+0")
    outputwindow.title("Output")
    label = tk.Label(outputwindow, text="Current Output of the Video")
    label.config(font=("Arial", 20))
    label.place(relx=0.05, rely=0.05)
    label1 = tk.Label(outputwindow, text="These are the classifications of emotions in the video")
    label.config(font=("Arial", 12))
    label1.place(relx=0.05, rely=0.09)
    playbutton = tk.Button(outputwindow, text="Play Video", command=outputvideoplay)
    playbutton.place(relx=0.05, rely=0.95)
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

    happinesslabel = tk.Label(outputwindow, text="Happy Emotions Timestamp")
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

    angertext = tk.Text(outputwindow, height=4, width=40)
    angertext.place(relx=0.05, rely=0.145)

    feartext = tk.Text(outputwindow, height=4, width=40)
    feartext.place(relx=0.05, rely=0.255)

    disgusttext = tk.Text(outputwindow, height=4, width=40)
    disgusttext.place(relx=0.05, rely=0.365)

    happinesstext = tk.Text(outputwindow, height=4, width=40)
    happinesstext.place(relx=0.05, rely=0.475)

    sadnesstext = tk.Text(outputwindow, height=4, width=40)
    sadnesstext.place(relx=0.05, rely=0.585)

    surprisetext = tk.Text(outputwindow, height=4, width=40)
    surprisetext.place(relx=0.05, rely=0.695)

    neutraltext = tk.Text(outputwindow, height=4, width=40)
    neutraltext.place(relx=0.05, rely=0.805)

    # input the time stamp of the video
    for i in range(5,len(filecontent),6):
        if str(filecontent[i])=="angry":
            angertext.insert(tk.END,filecontent[i-5]+" "+ filecontent[i-4]+" "+filecontent[i-3]+" "+filecontent[i-2]+ '\n')
        elif str(filecontent[i])=="disgust":
            disgusttext.insert(tk.END,filecontent[i-5]+" "+ filecontent[i-4]+" "+filecontent[i-3]+" "+filecontent[i-2]+'\n')
        elif str(filecontent[i])=="fear":
            feartext.insert(tk.END,filecontent[i-5]+" "+ filecontent[i-4]+" "+filecontent[i-3]+" "+filecontent[i-2]+'\n')
        elif str(filecontent[i])=="happy":
            happinesstext.insert(tk.END,filecontent[i-5]+" "+ filecontent[i-4]+" "+filecontent[i-3]+" "+filecontent[i-2]+ '\n')
        elif str(filecontent[i])=="sad":
            sadnesstext.insert(tk.END,filecontent[i-5]+" "+ filecontent[i-4]+" "+filecontent[i-3]+" "+filecontent[i-2] + '\n')
        elif str(filecontent[i])=="surprise":
            surprisetext.insert(tk.END,filecontent[i-5]+" "+ filecontent[i-4]+" "+filecontent[i-3]+" "+filecontent[i-2] + '\n')
        elif str(filecontent[i])=="neutral":
            neutraltext.insert(tk.END,filecontent[i-5]+" "+ filecontent[i-4]+" "+filecontent[i-3]+" "+filecontent[i-2] + '\n')

    angertext.config(state=tk.DISABLED)
    sadnesstext.config(state=tk.DISABLED)
    disgusttext.config(state=tk.DISABLED)
    feartext.config(state=tk.DISABLED)
    surprisetext.config(state=tk.DISABLED)
    neutraltext.config(state=tk.DISABLED)
    happinesstext.config(state=tk.DISABLED)

    outputlabel = tk.Label(outputwindow, text="Thank you for using our application! Please run the application again "
                                              "to process another video!", fg='red')
    outputlabel.config(font=("Arial", 10))
    outputlabel.place(relx=0.05, rely=0.90)
    outputwindow.mainloop()


def outputvideoplay():
    '''
    This function allows the operating system to browse through a default media player in our device. After locating
    the default media player, it will run the existing video that is in the directory.
    precondition: must be an mp4 file
    :param:none
    postcondition:none
    :return: video being played
    '''
    bvideo_name = "Tagged_video.mp4"  # This is your video file path
    os.system(bvideo_name)


def firstwindow():
    '''
    This is the main window of our application that shows the requirements of the video to be processed and our
    inspiration in working on these project. In the main window, there is an upload button where user can upload
    any video files to have their emotions being processed and a help button where it a guideline where user can
    follow it in order to use our software
    :return: the main window
    '''
    global mainwindow
    mainwindow = tk.Tk()
    mainwindow.title("Emotion Detection")
    mainwindow.geometry("600x700+450+50")

    button2 = tk.Button(mainwindow, text="Help", width=20, command=secondWindow)
    button2.place(relx=0.65, rely=0.8)

    load = Image.open('logo.png')
    render = ImageTk.PhotoImage(load)
    image = Label(image=render)
    image.image = render
    image.place(relx=0.57, rely=0.13)

    uploadbutton = tk.Button(mainwindow, text="Upload", command=browsebutton, width=20)
    uploadbutton.place(relx=0.65, rely=0.75)

    label = tk.Label(mainwindow, text="Emotion Detection")
    label.config(font=("Arial", 30))
    label.place(relx=0.25, rely=0.05)

    bg = mainwindow.cget("background")
    text2 = tk.Text(mainwindow, height=10, width=37, bg=bg, fg='blue', wrap=WORD)
    text2.place(relx=0.05, rely=0.14)
    text2.insert(tk.END, "Overview \nWe are Team4B, a group of 4 final year students in Monash University Malaysia "
                         "currently working on a project to detect macro emotions of a subject in a video.")
    text2.config(state=tk.DISABLED)

    text1 = tk.Text(mainwindow, height=20, width=42, wrap=WORD, fg='red')
    text1.insert(tk.END, "Requirement for Emotion Detection \n1. Subject of analysis shouldn't be more than 2 meters "
                         "away from the camera\n\n2. Can process up to 30 minutes of video\n\n 3. Video should "
                         "be recorded in good lighting\n\n4. Video should be recorded in decent quality\n\n5. "
                         "Only one subject should be present in the video\n\n6. Local machine must have an existing "
                         "video media player")
    text1.config(state=tk.DISABLED)
    text1.config(font=("Arial", 10))
    text1.place(relx=0.05, rely=0.4)

    mainwindow.mainloop()


if __name__ == '__main__':
    loginwindow()
