"""
draw_face_eyes.py

CSC 515 Foundations of Computer Vision
CSU Global

Module 3: Portfolio Milestone
Option #1: Drawing Functions in OpenCV

"It is time to think more about your upcoming Portfolio Project. 
In face and object detection, it is often useful to draw on the image. 
Perhaps you would like to put bounding boxes around features or use text to tag objects/people in the image. 

Use a camera to take a picture of yourself facing the frontal.  
In OpenCV, draw on the image a red bounding box for your eyes and a green circle around your face.  
Then tag the image with the text “this is me”.

Your submission should be one executable Python file."

Lincoln Quick
Date: June 26, 2025
"""
import cv2
import pathlib

# Paths to built-in Haar models shipped with OpenCV
cv_dir = pathlib.Path(cv2.__file__).parent / 'data'
FACE_CASCADE = cv_dir / 'haarcascade_frontalface_default.xml'
EYE_CASCADE = cv_dir / 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE))
eye_cascade = cv2.CascadeClassifier(str(EYE_CASCADE))

# Capture image from webcam
cap = cv2.VideoCapture(0) # default camera
ret, img = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Error capturing frame from webcam.")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the face

# Show the image
cv2.imshow("Original Capture", img)
cv2.waitKey(0)
cv2.imshow("Grayscale image", gray)
cv2.waitKey(0)

cv2.destroyAllWindows()
