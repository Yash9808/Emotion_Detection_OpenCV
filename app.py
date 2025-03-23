import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np

# Function to try opening the camera at different indices
def get_camera(cap_indices=[0, 1, 2]):
    for index in cap_indices:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        cap.release()
    return None

# Streamlit UI setup
st.title("Real-Time Emotion Detection with Webcam")

# Step 1: Prompt for user permission to use the camera
permission = st.radio(
    "This app will use your camera to detect facial expressions. Do you allow access to the camera?",
    ('Allow Camera', 'Deny')
)

# Step 2: Check user permission
if permission == 'Deny':
    st.error("❌ Camera access denied. The app will not work without camera access.")
else:
    # Try to get the camera feed from indices 0, 1, 2
    cap = get_camera([0, 1, 2])  # Testing indices 0, 1, 2

    if cap is None:
        st.error("❌ Camera Not Found! Please make sure your camera is connected and accessible.")
    else:
        st.success("✅ Camera Found!")
        
        # Capture frame-by-frame from the camera
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and analyze face in the captured frame
            try:
                # Use DeepFace for emotion recognition
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = analysis[0]['dominant_emotion']

                # Display emotion text on the frame
                cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print("Error analyzing face:", e)

            # Convert frame (BGR -> RGB) for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the resulting frame with the detected emotion
            st.image(frame_rgb, caption="Live Webcam Feed", use_column_width=True)

            # If the user presses 'q', exit the loop (this will not work in Streamlit, loop continues until manually stopped)
            
            # Release the capture and close the window when done
            cap.release()
            st.stop()
