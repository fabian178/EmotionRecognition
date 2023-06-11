
import cv2, glob, random, math, numpy as np, dlib, os
from sklearn.svm import SVC
from sklearn.externals import joblib

import pickle
from feature_extraction import *


'''
DONE
This script was used to read in the prepared datasets, extract the features and save them in a file for later loading.
Different SVM classifiers were then trained with this data and then also saved for later loading.
The SVM classifiers' accuracies were then tested on the test dataset.
'''



def mark_landmark_points(paths):
     
    
    #This function marks the dlib landmark points on selected images from the test dataset
    #this was used to mark the landmark points on the pixelated facial image from the test dataset in the thesis
    #dlib could not recognize all zoomed in faces, only some, the ones that werent recognized
    #were skipped and not included in the training
   
    #loop to read in and display all images
    for pic1 in paths:
        frame1 = cv2.imread(pic1) 
        try:
            dets = detector(frame1) #dets contains the face bounding box
        except:
            print('exception')
            continue
        if len(dets)== 0: #just in case nothing is detected
            continue
        for k,d in enumerate(dets):
            shape = predictor(frame1,d)
            print('number of points is: ' + str(len(shape.parts())))
            for indx,p in enumerate(shape.parts()): #from 0 to 67
                #print(p)
                pos = (p.x, p.y)
                #cv2.putText(frame,' ',pos,fontFace =cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            #fontScale = 0.4,color = (0,0,255))
                cv2.circle(frame1,pos,1,color = (0,255,255))        
        cv2.imshow('frame',frame1)
        key_pressed = cv2.waitKey(0)  #the pressed key is converted to its ASCII code
                
        #emotions = ["neutral", "anger", "disgust", "happiness", "sadness", "fear", "surprise"] #Emotion
        if str(unichr(key_pressed)) ==' ': #if o is pressed, continue to the next image, otherwise it stops
            continue
        else:
            exit()
    exit()
    
def create_SVM_model(classes, path_to_data, model, with_length, save_location='', load_model=''):
    '''
    THIS IS TO CREATE SVM MODELs FROM YOUR DATASETs
    enter your paths for your datasets into the names list. if you only have one dataset and, thus
    only one path, enter that path as a single entry into the list.
    
    the parameter train tells the function to train a new model or load a previously trained one.
    specify the path of a model to load in load_model
    '''
    
    #if we're loading a model, load it and skip the rest
    if not load_model=='':
        model = joblib.load(load_model)# this statement loads an existing model
        print('Finishded loading model: '+ load_model)
        return model
    
    #the rest here is for training new models
    
       
    print('currently extracting features for: ' + path_to_data)
    #prepare training data and labels
    training_data, training_labels = make_sets(classes, path_to_data, length=with_length, remove_sides=False)
    
    print('Number of training images: ' + str(len(training_labels))) #print out the number of training images
    
    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
   
    #these 2 statements save the extracted training feature and label arrays
    #pickle.dump(training_data, open('dataset//t.p', 'wb'))
    #pickle.dump(training_labels, open('dataset//l.p', 'wb'))
    
    #this section loads the saved training features that we saved using the above 2 lines
    #t = pickle.load(open('dataset//t.p', 'rb'))
    #l = pickle.load(open('dataset//l.p', 'rb'))
    #npar_train = np.array(t) #convert to numpy array to train the classifier with it
    #npar_trainlabs = np.array(l)#convert to numpy array to train the classifier with it
    
    
    print('Training new svm: ' + path_to_data)
    #this method trains the model, it takes in numpy arrays of 
    #the training features and corresponding labels.
    model.fit(npar_train, npar_trainlabs) 
    
    
    if not save_location=='':
        print('finished fitting the new svm, saving...')
        #save the trained svm model to the specified path
        joblib.dump(model,save_location)
        
    return model
    
def create_test_data_from_test_set(path_to_data, path_to_labels,classes, use_length):
    
    #this is for creating the test data
    #first read in the labels into a dict
    print('preparing prediciton database from the test set')
    #order the image names and their corresponding labels
    #the labels file is like this: IMAGE-NAME.EMOTION
    label_dict = {}
    label_file = open(path_to_labels, 'r')
    for line in label_file:
        num = line.split('.')[0] #get the image name
        emotion = line.split('.')[1] #get the corresponding emotion for this image
        emotion = emotion.replace('\n','') #take the newline characters out
        label_dict[num] = emotion #enter these two into a dictionary 
    
    
    all_pics = []
    for emot in classes:
        pics = glob.glob(path_to_data+'//%s//*' %emot)      
        all_pics.extend(pics)
   
    print('Number of test images: ' + str(len(all_pics)))
    
    label_list = []
    name_list =[]
    
    for pic in all_pics:
        name = pic.split('/')[-1].split('.')[0] #to get only the name of the image, without its file extension
        name_list.append(name)
        label_list.append(label_dict[name])
    
    p,labels = make_test_sets(all_pics, label_dict, leng=use_length)
    print('test dataset ready, length: ' + str(len(p)))
    
    
    #because the labels need to be given in integers, not string, we need to convert the label list to 
    #a list of integers corresponding to its appropriate label given by the class list
    #emotions = ["neutral", "anger", "disgust", "happiness", "sadness", "fear", "surprise"]
    number_labels = []
    for lb in labels:
        if lb=='neutral' or lb=='approval':
            number_labels.append(0)
        elif lb=='anger' or lb=='disapproval':
            number_labels.append(1)
        elif lb=='disgust' or lb == 'indifference':
            number_labels.append(2)
        elif lb=='happiness':
            number_labels.append(3)
        elif lb=='sadness':
            number_labels.append(4)
        elif lb=='fear':
            number_labels.append(5)
        elif lb=='surprise':
            number_labels.append(6)
              
    #you can save the extracted feature vectors and labels or load previously saved ones
    #this is useful if you have a large test set and it takes a long time to extract features and order it
    #every time
    
    #pickle.dump(p, open('dataset//test_features//close_faceonly//feature_list_no_neutral.p', 'wb'))
    #pickle.dump(number_labels, open('dataset//test_features//close_faceonly//test_labels_no_neutral.p', 'wb'))
    
    #p = pickle.load(open('dataset//feature_list_no_length.p', 'rb'))
    #number_labels = pickle.load(open('dataset//test_labels_no_length.p', 'rb'))
            
    return p,number_labels
    
    
def test_dataset(test_features, test_labels, model,application_testing=False, performance_testing=False, performance_testing_classes=[]):
         
    '''
    set application testing to true if you want to test accuracies with only approval/disapproval 
    when you have all 7 emotions as classes.
    set performane testing to true and specifiy which classes you want to include to do a push-the-limit test
    with all other emotions besides the ones that are specified removed from the test set.
    '''
   
    if performance_testing==True:
        #THIS SECTION IS ONLY FOR PERFORMANCE TESTING
        c = 0 #counter for index
        #the following for loop removes all entries from the test set that are not 
        #in the classes for performane testing
        listOfIndeces=[] 
        for lb in test_labels[:]: #loop over a copy of the list
           
            if lb in performance_testing_classes:
                pass
            else:
                test_labels.remove(lb)     #remove elements with index c
                listOfIndeces.append(c)      
            c+=1
            
        for i in sorted(listOfIndeces,reverse = True):
            test_features.pop(i) #remove the entries not used for performance testing
          
        # print the number of items removed
        print(str(len(test_labels)))#print the number of remaining items
        print('This many images were removed from the test set for performance testing: '+str(len(listOfIndeces)))
        #print the number of remaining items again from the feature list as 
        #confirmation that labels and features have the same number of items
        print('This many images are now in the test set for performance testing: '+str(len(test_features))) 
        #print the number of remaining items again from the feature list as 
        #confirmation that labels and features have the same number of items
        #PERFORMANC TESTING PREPARATION ENDS, TEST DATASET IS PREPARED
        
        np_features = np.array(test_features) #convert the lists to numpy arrays for the classifier
        np_labels = np.array(test_labels)
    
        #this method does the testing and returns a float representing the accuracy of the model
        resulting_accuracy = model.score(np_features,np_labels)
        print('The Accuracy of the Classifier under performance testing is: ' + resulting_accuracy)
        
        
        #this section does the prediction confidence
        #predict again, this time with prediction confidence
        #this method returns a list of all prediction probabilities for all classes
        prob = model.predict_proba(np_features)
        
        #need to sort through the proabibilities and select the highest one since the highest probability
        #is the class that was chosen by the classifier, then average all determined probabilities
        #to find the avergae prediction confidence of the model for the test set
        maxs = []
        for entry in prob:
            maxOfEntry = max(entry)
            maxs.append(maxOfEntry)
        avg = sum(maxs)/len(maxs)
        print('Prediction Confidence for model under performance testing is' + ' = ' + str(avg))
        
        return
    
    
    #application testing is testing only for approval/disapproval, ie classifying the emotion results
    #into approval/disapproval
    if application_testing==True:
        #convert features and labels into numpy arrays for the classifier
        np_features = np.array(test_features) 
        np_labels = np.array(test_labels)
        
        x = model.predict(np_features) #get the prediction results of the test set, 
        #then classify the results according to approval/disapproval/indifference
        #set up various counter
        approve=0
        disapprove=0
        na=0
        counter = 0
        wrong=0
        approval_list = ['happiness']
        disapproval_list = ['fear','disgust', 'anger', 'sadness']
        na_list = ['neutral', 'surprise']
        for result in x:
            #we need to convert the results to strings to compare it to the approval/disapproval lists
            #more intuitively
            if result==0:    
                 result1='neutral'
            elif result==1:
                result1='anger'
            elif result==2:
                result1='disgust'
            elif result==3:
                result1='happiness'
            elif result==4:
                result1='sadness'
            elif result==5:
                result1='fear'
            elif result==6:
                result1='surprise'
            
            
            if result1.lower() in approval_list and test_labels[counter] in approval_list:
                approve +=1
            elif result1.lower() in disapproval_list and test_labels[counter] in disapproval_list:
                disapprove+=1
            elif result1.lower() in na_list and test_labels[counter] in na_list:
                na+=1
            else:
                wrong+=1
                
            counter +=1
        
        total = wrong + na + approve + disapprove
        achieved_accuracy = (1-wrong)/total    
        print('Application testing accuracy = ' + str(achieved_accuracy))    
        
        #this section does the prediction confidence
        #predict again, this time with prediction confidence
        #this method returns a list of all prediction probabilities for all classes
        prob = model.predict_proba(np_features)
        
        #need to sort through the proabibilities and select the highest one since the highest probability
        #is the class that was chosen by the classifier, then average all determined probabilities
        #to find the avergae prediction confidence of the model for the test set
        maxs = []
        for entry in prob:
            maxOfEntry = max(entry)
            maxs.append(maxOfEntry)
        avg = sum(maxs)/len(maxs)
        print('Prediction Confidence for model with Application testing' + ' = ' + str(avg))
        return
    
          
    #if we are just doing a normal accuracy test, these lines will be executed
    #convert the data into numpy arrays for the classifier
    np_features = np.array(test_features)
    np_labels = np.array(test_labels)
    
    #this method does the testing and returns a float representing the accuracy of the model
    resulting_accuracy = model.score(np_features,np_labels)
    print('The Accuracy of the Classifier is: ' + str(resulting_accuracy))
             
    #this section does the prediction confidence
    #predict again, this time with prediction confidence
    #this method returns a list of all prediction probabilities for all classes
    prob = model.predict_proba(np_features)
    
    #need to sort through the proabibilities and select the highest one since the highest probability
    #is the class that was chosen by the classifier, then average all determined probabilities
    #to find the avergae prediction confidence of the model for the test set
    maxs = []
    c=0
    for entry in prob:
        c+=1
        maxOfEntry = max(entry)
        maxs.append(maxOfEntry)
    avg = sum(maxs)/len(maxs)
    print('Prediction Confidence for model:' + ' = ' + str(avg))
        
      
     
    

if __name__ == '__main__':
    
    #THESE ARE FOR THE SETUP FOR DLIB, ETC
    #enter your classes into this list, your subfolder names in your dataset must have the same names as your classes
    #names are case sensitive !!
    emotions = ["neutral", "anger", "disgust", "happiness", "sadness", "fear", "surprise"] #Emotion list
    #emotions = ['approval','disapproval', 'indifference']

    #setup for dlib
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # localized histogram equalization
    detector = dlib.get_frontal_face_detector()#the face detector
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #the feature detector for landmark points

   
    #THIS IS TO MARK LANDMARK POINTS ON IMAGES
    #this function marks the landmark on selected images and displayes those images.
    #enter a list with paths to your images to use it.
    #press the space button to go through the images, any other button to stop
    
    #mark_landmark_points(paths =['dataset//own/happiness//1.jpg','dataset//own/happiness//2.jpg']) 
    
    
    
    #THIS IS TO CREATE A MODEL
    #enter the paths to your datasets here to train the corresponding model
    path = 'dataset//own_resized_faceonly'
    save = 'pretrained_svms//linear_kernel//with_length//own_resized_faceonly_100'
    #create an en 'empty' model with a linear kernel in this case, it still needs to be trained
    #training is done in the create_SVM_model function
    #load = 'pretrained_svms//linear_kernel//with_length//own_resized_faceonly_100'
    model = SVC(kernel='linear', probability=True, tol=1e-3) 
    clf = create_SVM_model(emotions, path, model, with_length=True, save_location=save) #with this, the newly trained model is loaded into clf
    
    #TO LOAD A PREVIOUSLY CREATED MODEL
    #load = 'pretrained_svms//linear_kernel//with_length//own_resized_faceonly_100'
    #clf = create_SVM_model(emotions, path, model, with_length=True, load_model = load)
    
    
    #THIS IS TO TEST THE MODEL
    #this function extracts the features and prepares a new test dataset for the classifier
    path_to_test_dataset ='dataset//test_database_face_only'
    path_to_labels = 'test_labels.txt'
    test_features, test_labels = create_test_data_from_test_set(path_to_test_dataset,path_to_labels, emotions, use_length=True)
    
    test_dataset(test_features, test_labels, clf)
    
 
    
    