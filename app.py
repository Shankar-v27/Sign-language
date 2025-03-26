  
import streamlit as st  
import cv2  
import numpy as np  
import mediapipe as mp  
from model import KeyPointClassifier  
from model import PointHistoryClassifier  

# Initialize MediaPipe Hands module  
mp_hands = mp.solutions.hands  
hands = mp_hands.Hands()  

st.title("Real-Time Sign Language Detection")  

# Start video capture  
video = cv2.VideoCapture(0)  

# Streamlit UI  
frame_window = st.image([])  

while video.isOpened():  
    ret, frame = video.read()  
    if not ret:  
        st.write("Failed to capture video")  
        break  

    # Convert BGR to RGB  
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

    # Process frame with MediaPipe  
    results = hands.process(frame)  

    # Draw landmarks  
    if results.multi_hand_landmarks:  
        for hand_landmarks in results.multi_hand_landmarks:  
            for lm in hand_landmarks.landmark:  
                h, w, c = frame.shape  
                cx, cy = int(lm.x * w), int(lm.y * h)  
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  

    frame_window.image(frame)  

video.release()  
st.write("Stopped video capture.")  
