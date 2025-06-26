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

# Warm up by discarding first 29 frames
for _ in range(30):
    ret, img = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Error capturing frame from webcam.")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the face
faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5,
    minSize=(100,100))
if len(faces) == 0:
    raise RuntimeError("No face detected. Try again.")

# Bounding box for face
(x, y, w, h) = faces[0]

# Draw a green circle around the face
center = (x + w//2, y + h//2)
radius = int(0.9 * max(w, h) / 2) # Keep circle inside the box by scaling it
cv2.circle(img, center, radius, (0, 255, 0), thickness=2) # Green

# Detect the eyes
roi_gray = gray[y:y+h, x:x+w]
roi_img = img[y:y+h, x:x+w]

eyes = eye_cascade.detectMultiScale(
    roi_gray, 
    scaleFactor=1.05, 
    minNeighbors=2,
    minSize=(10,10))

# Draw red rectangles eyes detected
for (ex, ey, ew, eh) in eyes[:2]:
    cv2.rectangle(roi_img,
                  (ex, ey),
                  (ex+ew, ey+eh),
                  (0, 0, 255), 2)

# Show the image
cv2.imshow("Original Capture", img)
cv2.waitKey(0)
cv2.imshow("Grayscale image", gray)
cv2.waitKey(0)

cv2.destroyAllWindows()
