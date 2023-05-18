# 1 - Find the index of the working camera in your computer

## `$ python.exe list_available_cams.py`
## This script will list the available cams in your computer.

<br>
<br>

# 2 - Colect the images to create the dataset

## `$ python.exe collect_imgs.py`
<br>

## This will open a new window to capture the sign, try to moviment the hand on the camera to capture diferents angles and distances.
## After you finish, rename all the folders in inside './data' to the befitting signal that was captured. Ex: if the folder number 0 is the sign captured for 'A', rename the folder to 'A'.

<br>
<br>

# 3 - Create the dataset with the images collected

## `$ python.exe create_dataset.py`
<br>

## This code processes a directory containing hand images and extracts hand landmarks using the Mediapipe library. It saves the extracted hand landmarks along with their corresponding labels into a pickle file.

<br>
<br>

# 4 - Train the dataset which has been created

## `$ python.exe train_classifier.py`
<br>

## This code loads the hand landmarks data and labels, splits the data into training and testing sets, trains a random forest classifier on the training data, evaluates the model's performance on the testing data, and saves the trained model for future use.

<br>
<br>

# 5 - Finally, run the inference classifier to test

## `$ python.exe inference_classifier.py`
<br>

## The code captures video frames, detects and tracks hand landmarks in real-time, and predicts the corresponding hand gesture for each hand detected, displaying the results on the screen.


<br>
<br>
<br>
<br>
<br>
<br>

# Original repository from Computer vision developer, Copyright (c) 2023 Computer vision developer
## https://github.com/computervisioneng/sign-language-detector-python

Sign language detector with Python, OpenCV and Mediapipe!
<br>

[![Watch the video, Copyright (c) 2023 Computer vision developer](https://img.youtube.com/vi/MJCSjXepaAM/0.jpg)](https://www.youtube.com/watch?v=MJCSjXepaAM)
