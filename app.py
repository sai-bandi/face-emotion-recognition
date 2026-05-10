import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Real-Time Emotion Detector", layout="centered")

st.title("😊 Real-Time Emotion Detector")

# Load the pre-trained model
model = load_model("_mini_XCEPTION.102-0.66.hdf5", compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            prediction = model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
