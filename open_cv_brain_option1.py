"""
open_cv_brain_option1.py

CSC 515 Foundations of Computer Vision
CSU Global

Module 1: Portfolio Milestone
Option 1: Installing OpenCV 1

'For this milestone assignment, install OpenCV based on your specific operating system.  
Then, use OpenCV to complete the following:

- Write Python code to import the following image: brain image. Download brain image.
- Write Python code to display the image.
- Write Python code to write a copy of the image to any directory on your desktop.
Your submission should be one executable Python file.'

Lincoln Quick
June 9, 2025

"""
import cv2
import os

img = cv2.imread('shutterstock93075775--250.jpg')
path = '/Users/lincolnquick/Desktop'

if img is None: 
    print("Error: Could not load the 'brain' image. Check the filename/path.")
    exit(1)

# Read the brain image
cv2.imwrite('brain_copy.jpg', img)
# Display the image in a window
cv2.imshow('brain_window', img)
# Save a copy in the Desktop folder
cv2.imwrite(os.path.join(path, 'brain.jpg'), img)

# Wait then close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
