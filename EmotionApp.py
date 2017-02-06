import Tkinter
import tkMessageBox
from Tkinter import *
from tkFileDialog import askopenfilename
from sklearn.svm import SVC
from sklearn.externals import joblib
import cv2, math, numpy as np, dlib,threading
from PIL import ImageTk, Image
from cv2 import COLOR_BGR2GRAY
from timeit import Timer
import imghdr
from feature_extraction import *
import pyautogui

'''
DONE
This script is the GUI for real time emotion recognition from facial expressions.
external libraries that need to be installed:
opencv and all its dependencies, especially numpy
PIL which is an image library required for Tkinter (Pillow)
sklearn which has the implementation of the SVM classifier
dlib which contains the landmark recognition algorithm
ImageTk needs to be installed separately
pyautogui as described in the readme file
'''
#button types are used to determine behaviour when the callback is executed.
#The Checkbutton types are: 'test image', 'show face box', 'show video', 'show face', 'black and white', 'run',
#'run RT', 'show app_dis'
class MyCheckbutton: 
    def __init__(self,master, buttontext, typeOfButton, width = 20):
        self.buttonType = typeOfButton
        self.var = IntVar() #this is the status variable of the check button
        
        #self.c is the checkbutton that is created for this class
        
        #REMOVE ALL THESE IFS, SHOULDNT MAKE A DIFFERENCE
        
        #you can cutomize the buttons here, add different padding etc.
        if typeOfButton=='test image':
            self.c = Tkinter.Checkbutton(master, text = buttontext, variable = self.var,
                                command = self.cb, width = width)
         
        elif typeOfButton=='show app_dis':
            self.c = Tkinter.Checkbutton(master, text = buttontext, variable = self.var,
                                command = self.cb, width = width) 
            
        elif typeOfButton=='run':
            self.c = Tkinter.Checkbutton(master, text = buttontext, variable = self.var,
                                command = self.cb, width = width)
            
        
        elif typeOfButton=='run RT':
            self.c = Tkinter.Checkbutton(master, text = buttontext, variable = self.var,
                                command = self.cb, width = width)
          
            
        elif typeOfButton =='show face box':
            self.c = Tkinter.Checkbutton(master, text = buttontext, variable = self.var,
                                command = self.cb,pady=10, width = width)
         
        elif typeOfButton=='show video':
            self.c = Tkinter.Checkbutton(master, text = buttontext, variable = self.var,
                                command = self.cb, width = width)
           
        elif typeOfButton =='show face':
            self.c = Tkinter.Checkbutton(master, text = buttontext, variable = self.var,
                                command = self.cb, width = width)
           
        elif typeOfButton=='black and white':
            self.c = Tkinter.Checkbutton(master, text = buttontext, variable = self.var,
                                command = self.cb, width = width)
         
       
    def cb(self):
        
        #this is the callback method for the checkbuttons
        
        if self.buttonType =='show video':
            
            #when the buttons are selected or deselected, the appropirate buttons get disabled or enabled
            #eg. when the button to run real time is checked, disable the button for selecting a test image
            if self.var.get() ==1:
                app.disable_or_enable_test('disable') #these disable_or_enable methods are for en/disabling a bunc of buttons together instead of always having to type it out
                app.disable_or_enable_display('enable')
                app.test_image_path=''
                app.testImageSelection.config(text='Select a test image or run in real time\nwith live video.')
            else:
                if app.check_run_RT.var.get()==0:
                    app.button_select_test_image.c['state'] = 'normal'
                app.disable_or_enable_display('disable')
        elif self.buttonType=='show app_dis':
            pass # do nothing, it will only be checked if the buttons is checked or not in the main loop
        
        elif self.buttonType=='run':
            
            if self.var.get()==1:
                app.disable_or_enable_RT('disable')
                app.check_show_app_dis.c['state'] = 'normal'
                
            else:
                app.disable_or_enable_RT('enable') 
                #app.check_run.var.set(0)
                #app.check_run.c['state'] = 'disabled'
                app.check_show_app_dis.c['state'] = 'disabled'
                app.check_show_app_dis.var.set(0)
                
            
        elif self.buttonType=='run RT':
            
            if self.var.get() ==1:
                
                app.disable_or_enable_test('disable')
                app.test_image_path=''
                app.testImageSelection.config(text='Select a test image or run in real time\nwith live video.')
                app.check_show_app_dis.c['state'] = 'normal'
            else:
                app.check_show_app_dis.c['state'] = 'disabled'
                app.check_show_app_dis.var.set(0)
                if app.check_show_video.var.get()==0:
                    app.button_select_test_image.c['state'] = 'normal'
                    
        
#1280, 720 is default res of camera
#this is the class for the 'normal' rectangular buttons
class MyNormalbutton:
    def __init__(self, master, buttontext, typeOfButton):
        self.buttonType = typeOfButton
        #self.c is the actual button object
        
        #REMOVE ALL THESE IFS, SHOULDNT MAKE A DIFF
        
        self.c = Tkinter.Button(master, text = buttontext, command = self.cb)
        
        '''
        if typeOfButton=='select svms':
            self.c = Tkinter.Button(master, text = buttontext, command = self.cb)
            
        elif typeOfButton =='select image':
            self.c = Tkinter.Button(master, text = buttontext, command = self.cb)
            
        elif typeOfButton =='reset table':
            self.c = Tkinter.Button(master, text = buttontext,command = self.cb)
        '''
            
        #the callback for the normal buttons 
    def cb(self):
        
        if self.buttonType =='select svms': #when the select svms button was clicked
            
            old_svm_path = app.svm_path #store the svm_path before the new selection 
            #choose a model from previous model search path if it exists or default
            if not app.svm_path =='':
                app.svm_path=askopenfilename(initialdir=app.svm_path)#path from previous search
            else:
                app.svm_path=askopenfilename(initialdir='pretrained_svms') #default
                
            #if the file dialog is cancelled, an empty string is returned, we need to catch this 
            #and restore the variable to its old value, if there is an old value
            if app.svm_path=='' and old_svm_path != '': 
                app.svm_path=old_svm_path
                
            #take the last 3 directories of the path and show them in the label
            if not app.svm_path=='':
                l = app.svm_path.split('/')
                s = l[len(l)-3] + '/' + l[len(l)-2] +'/\n'+ l[len(l)-1]
                
                app.modelSelection.config(text=s)#update the label with the last 3 directories of the path
                
            
            #now try to load the selected svm model, if something invalid was selected, let the user know
            if not app.svm_path=='':
                try:
                    app.svm_model = joblib.load(app.svm_path) #load in the selected model
                    app.disable_or_enable_application('enable')
                    
                except:
                    if not app.svm_path =='': #if something wrong was selected and the dialog not cancelled
                        tkMessageBox.showwarning('Warning', 'Please select a valid SVM model.')
                    app.svm_path='' #reset the path to empty string
                    app.disable_or_enable_application('disable') #disable application setting (real time, test image, etc)
                    app.modelSelection.config(text = 'Please select a model.')#update the label again
            
        elif self.buttonType =='select image':
            
            if not app.test_image_path =='': #if a path was previously selected, use that for the dialog
                app.test_image_path=askopenfilename(initialdir=app.test_image_path)
            else:#if not, use the default path
                app.test_image_path=askopenfilename(initialdir='dataset//own')
            
            if not app.test_image_path=='': #if something was selected
                try:
                    a = imghdr.what(app.test_image_path) #test if it is a valid image file
                    if a == None:
                        raise Exception('invalid image format')
                    
                        
                    #take the last 3 directories of the selected path and display them in the appropriate label
                    l = app.test_image_path.split('/')
                    s = l[len(l)-3] + '/\n' + l[len(l)-2] +'/'+ l[len(l)-1]
                    app.testImageSelection.config(text=s)
                    
                        
                    app.check_run.c['state'] = 'normal'#enable the classify test image check button
                    app.disable_or_enable_display('enable')#enable the display section
                    app.image_scale.c['state'] = 'disabled' #but disable video scale since it doesnt apply
                       
                except:
                    tkMessageBox.showwarning('Warning', 'Please select a valid image file!')
                    app.test_image_path=''
                    
        elif self.buttonType=='reset table': #the reset table button just deletes all entries of the listbox
            app.results_listbox.delete(0, END)
        

class MyScale: #this is the class for the scale to adjust the video scale
    def __init__(self, master):
        self.var = IntVar()#the value of the scale
        self.is_pressed=False
        self.c = Tkinter.Scale(master, from_ = 25, to = 100, orient = Tkinter.HORIZONTAL)
        
        #bindings for when the scale is pressed and released
        self.c.bind("<ButtonPress>", self.on_press)
        self.c.bind("<ButtonRelease>", self.on_release)
        
    def on_press(self,event):
        self.is_pressed=True
        
    def on_release(self,event):
        self.is_pressed=False
    
    def getValue(self):
        return self.c.get()

'''
This is the main application.
First all buttons, values and variables are initialized and then the main loop for the video and test image started.
The main loop for the video and test images is started in a separate thread and due to python 2 not being
thread safe you have to keep moving the mouse to get the results from other running threads besides the main thread.
When classifying in real time or a test image, the approval/disapproval result, the classification result,
the confidence of that result  and the time it took to process are displayed. However, for each emotion, the result
is only displayed if a certain threshold in confidence is surpassed.

'''
class MyApp:
    def __init__(self, vs):
        self.vs = vs
        self.frame = None
        self.thread = None
        self.stopEvent = None
        
        #this is the svm classifier
        self.svm_model = SVC(kernel='linear', probability=True, tol=1e-3)
        
        self.root = Tkinter.Tk() #the root window
        
        #init all the subframes here and give them a position in the grid
        #the sticky value is the orientation, eg sticky = N means push the frame against the top
        self.svm_frame = Tkinter.Frame(self.root)
        self.svm_frame.grid(row=0,column=0, sticky = N)
        self.display_frame = Tkinter.Frame(self.root)
        self.display_frame.grid(row=1,column=0, sticky = N)
        self.select_frame = Tkinter.Frame(self.root)
        self.select_frame.grid(row=0, column=1, sticky = W)
        self.emotion_result_frame = Tkinter.Frame(self.root)
        self.emotion_result_frame.grid(row=2,column=1, sticky = N+W)
        self.listbox_frame = Tkinter.Frame(self.root, padx=10)
        self.listbox_frame.grid(row=1, column=2, sticky = W)
        self.description_frame = Tkinter.Frame(self.root)
        self.description_frame.grid(row=0, column=2, sticky = N)
       
    
        self.panel = None #the image panel in the middle of the window, this is updated throughout the main loop
        self.emotionLabel = None #the label for the resulting emotion, also updated throughout the main loop
        
        
        self.svm_path = ''#the path for the svm model
        
        
        #FOR DEBUGGING
        #self.svm_model=joblib.load('/Users/danielbozorgzadeh/Documents/pyWorkspace/opencv_BA/pretrained_svms/linear_kernel/no_length/own')
        #self.svm_path='/Users/danielbozorgzadeh/Documents/pyWorkspace/opencv_BA/pretrained_svms/linear_kernel/no_length/own'
        
        
        self.test_image_path = ''#the path to the test image
        self.video_scale = 0 #initial video scale
        self.resulting_emotion = '' #initial value for resulting emotion
        
        #create our buttons and listbox for results
        #listbox frame
        self.appLabel = Tkinter.Label(self.listbox_frame, text = 'Real Time Results', font = 'Helvetica 14 bold')
        self.appLabel.grid(row=0,column=0)
        self.results_listbox = Tkinter.Listbox(self.listbox_frame,height=15, width=30)#height is the number of elements 
        #that are displayed at one time, not pixels
        self.results_listbox.grid(row=2,column=0, sticky = W)
        self.reset_results_button = MyNormalbutton(self.listbox_frame,'reset table','reset table')
        self.reset_results_button.c.grid(row=1,column=0)
        
        #step 1 elements
        self.svm_label = Tkinter.Label(self.svm_frame, text = 'Step 1: Model Selection', font = 'Helvetica 14 bold')
        self.svm_label.grid(row=0,column=0,pady=(30,0))
        self.button_select_svm = MyNormalbutton(self.svm_frame, 'Select SVM model', 'select svms')
        self.button_select_svm.c.grid(row=1,column=0,pady=15)
        self.modelSelection = Tkinter.Label(self.svm_frame, text ='Please select a model.', width=30)
        self.modelSelection.grid(row=2,column=0)
        
        #step 2 elements
        self.appLabel = Tkinter.Label(self.select_frame, text = 'Step 2: Application Selection', font = 'Helvetica 14 bold')
        self.appLabel.grid(row=0,column=0,pady=(30,0))
        self.button_select_test_image = MyNormalbutton(self.select_frame, 'Select test image', 'select image')
        self.button_select_test_image.c.grid(row=2,column=0)
        self.testImageSelection = Tkinter.Label(self.select_frame, text ='Select a test image or run in real time\nwith live video.', width=30, 
                                                pady=10)
        self.testImageSelection.grid(row=1,column=0)
        self.check_run = MyCheckbutton(self.select_frame, 'classify test image','run')
        self.check_run.c.grid(row=3, column=0)
        self.check_run_RT = MyCheckbutton(self.select_frame, 'run in real time', 'run RT')
        self.check_run_RT.c.grid(row=4,column=0)
        self.check_show_video = MyCheckbutton(self.select_frame,'show live video', 'show video')
        self.check_show_video.c.grid(row=5,column=0 )
        self.check_show_app_dis = MyCheckbutton(self.select_frame,'approval/disapproval only', 'show app_dis')
        self.check_show_app_dis.c.grid(row=6,column=0)
        
        #display frame elements
        self.appLabel = Tkinter.Label(self.display_frame, text = 'Display Selection (optional)', font = 'Helvetica 14 bold')
        self.appLabel.grid(row=0,column=0)
        self.check_show_face_bounding_box = MyCheckbutton(self.display_frame, 'show face bounding\n box and points', 'show face box')
        self.check_show_face_bounding_box.c.grid(row=1,column=0 )
        self.check_black_and_white = MyCheckbutton(self.display_frame,'black & white','black and white')
        self.check_black_and_white.c.grid(row=2,column=0)
        self.check_show_only_face = MyCheckbutton(self.display_frame,'show only the face', 'show face')
        self.check_show_only_face.c.grid(row=3,column=0)
        self.scaleLabel = Tkinter.Label(self.display_frame, text ='scale of the video (%) : ',pady = 5)
        self.scaleLabel.grid(row=4,column=0)
        self.image_scale = MyScale(self.display_frame)
        self.image_scale.c.grid(row=5, column=0)
        
        
        #descrition frame elements
        self.descriptionTitle = Tkinter.Label(self.description_frame, text = 'Description', font = 'Helvetica 14 bold')
        self.descriptionTitle.grid(row=0,column=0, sticky = W,pady=(30,0))
        self.descriptionLabel = Tkinter.Label(self.description_frame, justify = LEFT, wraplength = 240, 
                                              text = 'This application will classify emotions from facial expressions from live video input or from a selected image. Follow step 1 to choose a model and step 2 to use a test image or do real time classification. The optional display section allows to display additional information or show modifications on the video or image input.')
                                              #text = 'This application will classify emotions from\nfacial expressions from live video\ninput or from a selected image.\nFollow step 1 to choose a model\nand step 2 to use a test image\nor do real time classification. The\noptional display section allows to display\nadditional information or show\nmodifications on the video or image input.')
        self.descriptionLabel.grid(row=1,column=0)
        
        
        #emotion result frame, under the image/video window in the middle
        self.emotionLabel = Tkinter.Label(self.emotion_result_frame, text ='Emotion: ', width = 45
                                          ,anchor=W)
        self.emotionLabel.grid(row=0,column=0)
        
        
        #do initial button settings here, disable step 2 and display at first
        self.disable_or_enable_application('disable')
        self.disable_or_enable_display('disable')
        
        #this starts the camera in a separate thread and loops to read input
        
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target = self.videoLoop, args=())
        self.thread.daemon=True
        self.thread.start()
        #start the thread here to move the mouse so the GUI updates properly after every frame
        self.thread1 = threading.Thread(target=self.moveMouse, args=())
        self.thread1.daemon=True
        self.thread1.start()
        
        #set callback to handle window close
        self.root.wm_title('Realtime Emotion Recognition GUI - Guja Bozorgzadeh')
        self.root.wm_protocol('WM_DELETE_WINDOW', self.onClose)
        
        self.detector = dlib.get_frontal_face_detector()#the dlib face detector
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #the feature detector
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # localized histogram equalizer
        
    '''  
    we need this method to have the app run smoothly. Because python2 is not thrad-safe the GUI is not updated
    properly after every frame. It only updates after movong the mouse. So we move the mouse automatically
    to make it update properly.
    '''
    def moveMouse(self):
        while(1):
            pyautogui.moveRel(1,0,0)
            pyautogui.moveRel(-1,0,0)
        
    def videoLoop(self):
        
        scale_of_vid = self.image_scale.getValue() #this is the initial value for the scale
        
        clahe = self.clahe
        detector = self.detector
        predictor = self.predictor
        
        #this is all in a try block because due to the nature of python 2 and the Tkiner library, 
        #when exiting the application, an exception is thrown. To avoid disturbance, it is caught and ignored.
        try:
            while not self.stopEvent.is_set(): #as long as the stop event for the thread has not been set, keep looping
                
               
                
                '''
                first check if a test image has been selected, if it has, do display, classification etc.
                on it. Update the panel to display whatever test image has been selected. Also, just in case
                check that showing live video is not selected.
                '''
                if self.test_image_path !='' and self.check_show_video.var.get() ==0:
                    if not self.test_image_path =='':
                        
                        
                        try:
                            img1 = cv2.imread(self.test_image_path) #read in the selected test image
                            #make a copy, the original is used for feature extraction and classification, 
                            #the copy is edited to be displayed in the panel. eg transformed to black&white,
                            #only the face showing etc.
                            original_image = img1.copy() 
                            
                            img1
                            #if black&white checked, convert it
                            if self.check_black_and_white.var.get() ==1:
                                try:
                                    img1 = cv2.cvtColor(img1,COLOR_BGR2GRAY)#will throw excpetion if it alrdy is gray
                                
                                except:
                                    pass #if it already is gray, pass, can't be any other exception since
                                            #all others were already caught when selecting the test image
                                
                            
                            try:#if it is a colored image, this will modify it correctly for Tkinter
                                #if not, simply keep going
                                #opencv reads in images in a bgr colorspace, tkinter needs the rgb color space
                                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                            except:
                                print('wasnt a colored image')
                                
                            #now do face detection, if display bounding box is selected
                            if self.check_show_face_bounding_box.var.get()==1:
                                dets = detector(img1) #this contains the faces' bounding boxes
                                for k,d in enumerate(dets): #enumerate over all face detections
                                    shape = predictor(img1,d) #this gets the 68 landmark points
                                    
                                    #if not show face only selected, draw a rectangle around the detected face
                                    if not self.check_show_only_face.var.get()==1:
                                        cv2.rectangle(img1,(d.right(),d.top()),(d.left(),d.bottom()),(0, 255, 0),2)

                                    
                                    for indx,p in enumerate(shape.parts()): #from 0 to 67 points
                                        pos = (p.x, p.y)
                                        cv2.circle(img1,pos,1,color = (0,255,255)) #mark the points on the image
                                        
                                    
                                               
                            #now cut the face out if only the face should be shown
                            if self.check_show_only_face.var.get()==1:
                                dets = detector(img1) #this gets all the detected faces' bounding box
                                for k,d in enumerate(dets):
                                    img1 = img1[d.top():d.bottom(),d.left():d.right()]
                                    break #do it for only 1 face 

                            #convert the edited image into tkinter format
                            img2 = Image.fromarray(img1)
                            img2 = ImageTk.PhotoImage(img2)
                            if self.panel==None: #update the panel and load the image into it
                                self.panel = Tkinter.Label(self.root, image = img2)
                                self.panel.grid(row=1,column=1)
                                self.panel.image = img2
                            
                            else: #if it alrdy initialized, simply update it  
                                self.panel.configure(image = img2)
                                self.panel.image = img2
                            
                            #if the check button to classify the image is checked:
                            #fyi, the classify button for the test image is called check_run,
                            #the check button for real time is called check_run_RT
                            if self.check_run.var.get()==1:
                                t = Timer(lambda:self.getAndDisplayResult(original_image))#pass the image into the feature extractor and classifier
                                #time the function execution
                                tt = t.timeit(number=1)
                                #update the emotion label to display the results
                                self.emotionLabel.config(text = self.emotionLabel['text'] + ', response time: %0.3f' %tt)
                            else:
                                self.emotionLabel.config(text = 'Emotion: ') 
                                
                                
                        except: #if any excpetion is thrown, reset the test image path and continue
                            self.test_image_path=''
                
                
                #if a test image has been selected, don't continue this loop to the bottom, start over again
                #since the rest is for real time video
                if self.test_image_path != '':
                    continue
                
                
                
                ret, frame1 = self.vs.read()#read the frame from the camera
                if ret == True:
                    #same concept: do modifications checked in the display section on a copy of the frame
                    #and use the original for feature extraction and classification
                    original_frame = frame1.copy()
                    
                    #1280, 720 is default resolution of the camera
                    if self.check_show_video.var.get()==1: #if show video is checked, scale the image to the
                        #selected scale multiplied by the max size: 711,400
                        #max size was determined to be the size that would still fit on the screen
                        xScale = int(711*scale_of_vid)
                        yScale = int(400*scale_of_vid)
                        if self.image_scale.is_pressed==False:#only scale once the scale is not pressed anymore
                            scale_of_video = float(self.image_scale.getValue()/100.0)
                            xScale = int(711*scale_of_video)
                            yScale = int(400*scale_of_video)
                            scale_of_vid=scale_of_video
                        frame1 = cv2.resize(frame1, (xScale,yScale))
                        
                        
                        #check if only want the face
                        if self.check_show_only_face.var.get()==1:
                            dets = detector(frame1) #this gives all the detected faces' bounding boxes
                            for k,d in enumerate(dets):#d is the bounding box, k is the index of the face in the list
                                frame1 = frame1[d.top():d.bottom(),d.left():d.right()]  
                                try:
                                    cv2.resize(frame1,(100,100)) #resize it to 100x100
                                except:
                                    pass#if something went wrong, just skip it and wait for the next frame
                                break#only do one face
                            
                        #convert to gray here if desired
                        if self.check_black_and_white.var.get() ==1:
                            try:       
                                frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                            except:
                                print('something went wrong converting camera input to gray')
                                pass  
                        
                        if self.check_show_face_bounding_box.var.get()==1:#if show face bounding box and landmark points is checked
                            dets = detector(frame1) #this contains the faces' bounding boxes
                            for k,d in enumerate(dets):#d is the face bounding box
                                shape = predictor(frame1,d)#this gives the landmark points
                                #draw a rectangle around the face
                                cv2.rectangle(frame1,(d.right(),d.top()),(d.left(),d.bottom()), (0, 255, 0),2)
                               
                                
                                for indx,p in enumerate(shape.parts()): #landmark points from 0 to 67
                                    #mark the 68 landmark points on the image
                                    pos = (p.x, p.y)
                                    cv2.circle(frame1,pos,1,color = (0,255,255))

                        #process the image for tkinter
                        img = Image.fromarray(frame1)
                        img = ImageTk.PhotoImage(img) #convert it to a Tkinter image
                        #if the panel hasnt been created yet, initialize it
                        if self.panel==None:
                            self.panel = Tkinter.Label(self.root, image = img)
                            self.panel.grid(row=1,column=1)
                            self.panel.image = img
                            
                        else: #if it has already been created simply update it  
                            self.panel.configure(image = img)
                            self.panel.image = img
                    else: #if show video is not selected
                        if self.panel==None:
                            pass #do nothing if the panel has not yet been created
                            
                        else:  
                            self.panel.configure(image = '') #reset the panel to show nothing
                            self.panel.image = ''
                        
                    
                    #if real time is checked
                    if self.check_run_RT.var.get() ==1:
                        #pass the frame into the feature extractor and time the function's execution
                        #this may cause excpetions sometimes, if an exception happens just continue normally
                        try:
                            t = Timer(lambda:self.getAndDisplayResult(original_frame))
                            tt = t.timeit(number=1)
                        except:
                            continue
                        #update the label to display results
                        self.emotionLabel.config(text = self.emotionLabel['text'] + ', response time: %0.3f' %tt)
                        
                        #if an emotion was detected the word 'probability' will be in the emotionLabel's text
                        #use this to determine whether an emotion was detected
                        #so if an emotion was detected enter the results into the end of the listbox and select the last element
                        if 'probability' in self.emotionLabel['text']:
                            result_text = str(self.results_listbox.size()) +'. ' +self.emotionLabel['text']
                            self.results_listbox.selection_clear('end')
                            self.results_listbox.insert(END, result_text)
                            self.results_listbox.see('end')
                            self.results_listbox.selection_set('end')
                   
                    else:
                         self.emotionLabel.config(text = 'Emotion: ')  #if no real time is running, reset the label to show nothing 
                        
                    cv2.waitKey(1) #cv2 function to get the next frame
                
                else:
                    break
                    print('frame couldnt be read, exiting')
                    
                    
            
        #this is the exception thrown when exiting the application, simply ignore it and let the app shut down
        except Exception as e: 
            pass
            
        
    
    def getAndDisplayResult(self, image):
        #this function does the classification. The original test image or video frame is passed into here
        
        #cut out face, convert to gray and do histogram equalization, then pass to extract features
        dets = app.detector(image)#get the faces' bounding boxes
        for k,d in enumerate(dets):
            image = image[d.top():d.bottom(),d.left():d.right()] #cut out the face
            break #only do one detected face
        
        #the svm models with no length contain the string 'no_length' in their name
        #so if a model does not contain 'no_length' in its name, it is with length
        if not 'no_length' in app.svm_path: #if using length feature, rescale to 100x100
            try:
                cv2.resize(image,(100,100)) 
            except:
                print('error resizing face in getanddisplayresults method')
                return #stop execution of this method
        
          
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
        clahe_image = app.clahe.apply(gray)# do contrast limited histogram equalization
       
        
        if 'no_length' in app.svm_path:
            extracted_features = get_landmarks(clahe_image, use_length=False)#feature extraction for no length
        else:
            extracted_features = get_landmarks(clahe_image, use_length=True)#feature extraction for with_length
        
        if extracted_features=='error': #if no features could be extracted, don't continue the method
            app.emotionLabel.configure(text = 'No face could be recognized') #update the label
            return
            
        
        #if no model is selected for some reason, don't continue the method and update the label
        if self.svm_path=='':
            app.emotionLabel.configure(text = 'Please select a model.')
            return
        
        try:
            data = np.asarray(extracted_features) #convert the features into a numpy array
            #data.reshape is called to make processing more efficient
            pred = self.svm_model.predict_proba(data.reshape(1,-1))#this does the prediction
            #returns a list of confidence values between 0 and 1 for all classes in the model
        except:
            #if an error occurs here, skip this image and wait for the next one
            return
            
        
        
        #this next section gets the max confidence value and assigns it an emotion
        counter =0
        highest = 0.0
        emotion_name = ''
        predicted_emotion = 0
        for p in pred[0]:
            if p > highest:
                highest = p
                predicted_emotion=counter
            counter +=1
        #convert it to %
        highest *=100.0
        
         #see what emotion name the predicted class represents,
         #predticions from the svm model only return an integer corresponding to its index in the list of classes
         #that was into it
        if len(pred[0]) ==7:
            emotion_name = emotions[predicted_emotion]
        else:
            emotion_name=emotions1[predicted_emotion]
       
        
        #each confidence must surpass a certain threshold value to be able to show the emotion as detected
        #this threshold for each emotion was determined by experimentation and was selected as the lowest value 
        #at which the model could still relatively confidently tell the emotion.
        #if the confidence did not surpass this value, the emotion was not clearly detected
        noEmotion = False
        te = 'Emotion could not be clearly recognized'
        
        threshold_for_current_emotion = thresholds[predicted_emotion]
        if highest < threshold_for_current_emotion:
            app.emotionLabel.configure(text = te)
            noEmotion = True
        
       
        #if our model is only approva, disapproval, neutral, this suffices
        endResult = emotion_name
        
        
        #if our model is with emotion classes, these if cases will overwrite emotion_name to the detected emotion
        #now to gorup the detected emotions into approval or disapproval or indifference
        #approval, disapproval, indifference are defined in the main method 
        if emotion_name in approve:
            endResult = 'Approval'
        elif emotion_name in disapprove:
            endResult = 'Disapproval'
        elif emotion_name in indifference:
            endResult ='Indifference'
        
        
        #if the length of the classes list is 7, we have emotions, if not we have approval/disapp..
        show_approval_or_disapproval = 0
        if not len(pred[0])==7:
            show_approval_or_disapproval=1
            
        #if an emotion was detected, update the label with the results
        if noEmotion==False:
            if self.check_show_app_dis.var.get()==1 or show_approval_or_disapproval==1:
                #show only approval/disapproval
                app.emotionLabel.configure(text = endResult + ': probability: %.2f'  %highest + '%')
            else:
                app.emotionLabel.configure(text = endResult + ': '+ emotion_name + ', probability: %.2f'  %highest + '%')
                
                
    def onClose(self):
        #set the stop event to close the camera thread, cleanup, etc

        print('closing...')
        self.stopEvent.set()
        self.vs.release()               
        self.root.destroy()
        exit()
        
    #these dis_or_enable_functions are just to disable or enable a group of buttons depending on what is selected in the program
    def disable_or_enable_SVM_select(self,choice):
        if choice =='disable':
            self.button_select_svm.c['state'] = 'disabled'
        elif choice == 'enable':
            self.button_select_svm.c['state'] = 'normal'
        
    def disable_or_enable_RT(self,choice):
        if choice == 'disable':
            self.check_run_RT.c['state'] = 'disabled'
            self.check_show_video.c['state'] = 'disabled'
        elif choice =='enable':
            self.check_run_RT.c['state'] = 'normal'
            self.check_show_video.c['state'] = 'normal'
        
    def disable_or_enable_test(self,choice):
        if choice == 'disable':
            self.check_run.c['state'] = 'disabled'
            #self.check_use_test_image.c['state'] = 'disabled'
            self.button_select_test_image.c['state'] = 'disabled'
        elif choice =='enable':
            pass
            #self.check_run.c['state'] = 'normal'
            #self.check_use_test_image.c['state'] = 'normal'
         
    
    def disable_or_enable_display(self,choice):
        
        if choice =='disable':
            self.check_show_face_bounding_box.c['state'] = 'disabled'
            self.check_black_and_white.c['state'] = 'disabled'
            self.check_show_only_face.c['state'] = 'disabled'
            self.image_scale.c['state']='disabled'
            
        elif choice =='enable':
            self.check_show_face_bounding_box.c['state'] = 'normal'
            self.check_black_and_white.c['state'] = 'normal'
            self.check_show_only_face.c['state'] = 'normal'
            self.image_scale.c['state']='normal'
            
            
    def disable_or_enable_application(self,choice):
        
        if choice =='disable':
            self.button_select_test_image.c['state'] = 'disabled'
            #self.check_use_test_image.c['state'] = 'disabled'
            self.check_run_RT.c['state'] = 'disabled'
            self.check_show_video.c['state'] = 'disabled'
            self.check_run.c['state'] = 'disabled'
            self.check_show_app_dis.c['state']='disabled'
        elif choice == 'enable':
            self.button_select_test_image.c['state'] = 'normal'
            #self.check_use_test_image.c['state'] = 'normal'
            self.check_run_RT.c['state'] = 'normal'
            self.check_show_video.c['state'] = 'normal'
            #self.check_run.c['state'] = 'normal'
    

if __name__=='__main__':
    
    
    #initialize these lists and input your classes and thresholds
    emotions = ["neutral", "anger", "disgust", "happiness", "sadness", "fear", "surprise"] #Emotion list
    emotions1 = ['approval', 'disapproval', 'indifference']
    
    thresholds = [76,74,77,73,80,78,81]
    approve = ['happiness']
    disapprove = ['anger', 'disgust', 'sadness', 'fear']
    indifference = ['neutral', 'surprise']
                
    
    #initialize a MyApp object and pass the cv2.videocapture object into it
    app = MyApp(cv2.VideoCapture(0))
    app.root.mainloop()#start the tkinter main loop for the GUI
