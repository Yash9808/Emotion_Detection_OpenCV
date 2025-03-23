import cv2
import requests
import numpy as np
from deepface import DeepFace
import os
import streamlit as st

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"


# Step 1: Allow user permission to use the camera
st.title("Face Detection and Emotion Recognition")

st.write("This app will use your camera to detect facial expressions.")
permission = st.radio("Do you allow access to the camera?", ('Allow Camera', 'Deny'))

if permission == 'Deny':
    st.error("❌ Camera access denied. The app will not work without camera access.")
    st.stop()

# Step 2: Download Haar Cascade for face detection
cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
cascade_path = "haarcascade_frontalface_default.xml"

if not os.path.exists(cascade_path):  # Download only if not already present
    response = requests.get(cascade_url)
    with open(cascade_path, "wb") as file:
        file.write(response.content)

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cascade_path)

# Step 3: Use OpenCV to access the camera
stframe = st.empty()  # Streamlit placeholder to update the frame

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("❌ Camera not found.")
    st.stop()

# Create a button to stop the loop
if st.button('Stop'):
    st.stop()

# Step 4: Continuous stream of frames for face and emotion detection
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Unable to fetch frame from camera.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If faces are found, draw rectangles around them and analyze expression
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face from the frame for expression recognition
        face_img = frame[y:y + h, x:x + w]

        # Use DeepFace to analyze the face expression
        try:
            analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']

            # Display the detected emotion on the frame
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            st.error(f"Error analyzing face: {str(e)}")

    # Convert frame to RGB for displaying in Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Update the displayed image in the Streamlit app
    stframe.image(frame, channels="RGB", use_column_width=True)

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
