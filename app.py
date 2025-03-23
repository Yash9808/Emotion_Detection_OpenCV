import cv2
import requests
import numpy as np
from deepface import DeepFace
import os
import PySimpleGUI as sg


# Step 1: Prompt for user permission to use the camera
layout = [
    [sg.Text("This app will use your camera to detect facial expressions.")],
    [sg.Button("Allow Camera"), sg.Button("Deny")]
]

window = sg.Window("Camera Permission", layout)
event, values = window.read()

# If the user denies permission, close the app
if event == "Deny" or event == sg.WIN_CLOSED:
    window.close()
    exit()

window.close()

# Step 2: Download Haar Cascade for face detection
cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
cascade_path = "haarcascade_frontalface_default.xml"

if not os.path.exists(cascade_path):  # Download only if not already present
    response = requests.get(cascade_url)
    with open(cascade_path, "wb") as file:
        file.write(response.content)

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cascade_path)

# Step 3: Open the computer's camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()
    if not ret:
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
            print("Error analyzing face:", e)

    # Show the resulting frame with face and emotion detection
    cv2.imshow('Face Detection and Emotion Recognition', frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
