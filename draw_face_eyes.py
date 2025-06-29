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

# Warm up by discarding first 29 frames for auto-exposure
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

# If no face, throw error and exit gracefully
if len(faces) == 0:
    raise RuntimeError("No face detected. Try again.")

# Bounding box for face
(x, y, w, h) = faces[0]

# Draw a green circle around the face
center = (x + w//2, y + h//2)
radius = int(0.9 * max(w, h) / 2) # Keep circle inside the box by scaling it
cv2.circle(img, center, radius, (0, 255, 0), thickness=2) # Green

# Detect the eyes within detected-face region
# crop a smaller region of interest (roi) to speed detection and reduce false positives
# from nostrils or corner of mouth, etc.
roi_gray = gray[y:y+h, x:x+w]
roi_img = img[y:y+h, x:x+w]

eyes = eye_cascade.detectMultiScale(
    roi_gray, 
    scaleFactor=1.05, 
    minNeighbors=5,
    minSize=(10,10))

# Draw red rectangles eyes detected
for (ex, ey, ew, eh) in eyes[:2]:
    cv2.rectangle(roi_img,
                  (ex, ey),
                  (ex+ew, ey+eh),
                  (0, 0, 255), 2)
    
# Add text label
text = "this is me"
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thick = 2
text_size, _ = cv2.getTextSize(text, font, 1, 2)

# Get text dimensions
(text_w, text_h), _ = cv2.getTextSize(text, font, scale, thick)

# center text between face circle and 40 px below
text_x = center[0] - text_w // 2 # center text horizontally
text_y = center[1] + radius + text_h + 40 # vertically place text 40px below face box

cv2.putText(img, 
            text, 
            (text_x, text_y), 
            font, 
            scale, 
            (255, 255, 255), 
            thick, 
            cv2.LINE_AA)


# Show the image
cv2.imshow("Annotated image", img)
cv2.waitKey(0)

# Save the image
cv2.imwrite("me_annotated.jpg", img)

# Close all windows and free up resources
cv2.destroyAllWindows()
