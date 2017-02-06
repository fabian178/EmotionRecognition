Libraries required:

1. OpenCV and all its dependencies (they are installed along with OpenCV):
	pip install opencv-python

	test the installation by running 'import cv2' in a python shell

2. sklearn:
	pip install -U scikit-learn

3. dlib:
	first install cmake:
	sudo add-apt-repository ppa:george-edison55/cmake-3.x
	apt-get update
	apt-get install cmake

	then install boost python:
	apt-get install libboost-all-dev

	then dlib:
	pip install dlib

4. PIL:
	pip install pillow

5. ImageTk:
	apt-get install python-imaging-tk

6. pyautogui:
	pip install pyobjc-core
	pip install pyobjc
	pip install pyautogui




To use the GUI: no further setup required, simply run the python script


To create a new model, test a model or mark landmark points on an image, first define these variables for setup:
	emotions = ["neutral", "anger", "disgust", "happiness", "sadness", "fear", "surprise"] #Emotion class list
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # localized histogram equalization
	detector = dlib.get_frontal_face_detector() #the face detector
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


To create a new model use the script 'model_creation_and_testing.py':
	-first create an SVM model object:
	model = SVC(kernel='linear', probability=True, tol=1e-3) 
	-then call this functon as follows:
	model = create_SVM_model(classes, path_to_data, model, with_length, save_location='', load_model='')

	-pass in the classes your model should have, the path to the dataset, the model object, True or False for with_length, optional: save location for the model or load location to load a previously trained model




To test a model:

	-first create a set of prediction data and labels:
	test_features, test_labels = create_test_data_from_test_set(path_to_data, path_to_labels,classes, use_length)

	Pass in the path to your test images dataset (they must be grouped into subfolders for each emotion), the path to the label file, the classes, and True or False for use_length. The label file must be in this format:
	1.neutral
	2.happiness
	3.neutral
	...

	-then call this function to carry out the test, results are printed to the console
	test_dataset(test_features, test_labels, model,application_testing=False, performance_testing=False, performance_testing_classes=[])

	Pass in the test_features and test_labels, the trained model, optional application_testing True/False to test only for approval/disapproval, performance_testing True/False to do a pushthe limit test, and enter the classes to use for the push the limit test in the list in performance_testing_classes.




To mark landmark points on an image:
	-call this function and pass the paths to the images to be marked. Press the space bar to go to the next image or any other button to exit.

	mark_landmark_points(paths)
	



