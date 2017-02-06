import cv2, glob, random, math, numpy as np, dlib, os
from sklearn.svm import SVC
from sklearn.externals import joblib

import pickle


'''
This script defines the functions for feature extraction and image gathering and is imported
into the other scripts.
'''
#setup for dlib
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # localized histogram equalization
detector = dlib.get_frontal_face_detector()#the face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #the feature detector for landmark points


def make_test_sets(image_path_list,label_dict, leng): 
    #this returns the prediction data and prediction labels from the test dataset
    #the parameter leng indicates if length is to be used or not
    
    prediction_data = []
    prediction_labels = []
    
    for item in image_path_list:
        
        image = cv2.imread(item)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            print('exception when making test set')
            continue
        clahe_image = clahe.apply(gray)
        
        landmarks_vectorised = get_landmarks(clahe_image, use_length=leng)
        
        if landmarks_vectorised == "error":
            continue
        else:
            prediction_data.append(landmarks_vectorised)
            prediction_labels.append(label_dict[item.split('/')[-1].split('.')[0]])
    
    return prediction_data, prediction_labels   
 
 #this returns a list of strings with the file directories for the training images to be read  
def get_training_files(emotion,path_for_data, remove_sides): 
    files = glob.glob(path_for_data + "//%s//*" %emotion)
    
    if remove_sides == True:
        print('sides true')
    if remove_sides == True:
        print('sides being skipped')
        for f in files[:]:
            #take out side images here
            if 'right' in f or 'left' in f:
                files.remove(f) #remove pic with side
    
    random.shuffle(files)
    training = files[:int(len(files))] 
    
    return training

#this returns training data and training labels 
def make_sets(list_of_classes, path_for_data, length, remove_sides=False): 
    training_data = []
    training_labels = []
    
    for class_ in list_of_classes:
        
        training = get_training_files(class_, path_for_data,remove_sides)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            
            landmarks_vectorised = get_landmarks(clahe_image, use_length = length)
            
                
            if landmarks_vectorised == "error":
                continue
            else:
                training_data.append(landmarks_vectorised) #append image array to training data list
                training_labels.append(list_of_classes.index(class_))

    return training_data, training_labels   

def get_landmarks(image, use_length):#feature extraction with or without length depending on the length variable
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            
        xmean = np.mean(xlist) #Get the mean of both axes to determine centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist] #get distance between each point and the central point in both axes
        ycentral = [(y-ymean) for y in ylist]

        if xlist[26] == xlist[29]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            if use_length==True: #include length features if length is True
                landmarks_vectorised.append(x)
                landmarks_vectorised.append(y)
                landmarks_vectorised.append(dist)
                
            anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
            landmarks_vectorised.append(anglerelative)
            
           

    if len(detections) < 1: 
        landmarks_vectorised = "error"
    return landmarks_vectorised

