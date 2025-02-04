# -*- coding: utf-8 -*-
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import sys
import os

if sys.version_info[0] >= 3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Use os.path to handle file paths more robustly
base_path = r"C:\Users\balde\Desktop\ML\Project\final"
face_classifier_path = os.path.join(base_path, 'haarcascade_frontalface_default.xml')
model_path = os.path.join(base_path, 'model.h5')

# Import other necessary libraries
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

# Use os.path.normpath to ensure correct path formatting
face_classifier = cv2.CascadeClassifier(os.path.normpath(face_classifier_path))
classifier = load_model(os.path.normpath(model_path))

# Print model input shape for verification
print("Model Input Shape:", classifier.input_shape)

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x,y,w,h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
        
        # Extract the region of interest (ROI)
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize ROI to 48x48
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Check if the ROI is not empty
        if np.sum([roi_gray]) != 0:
            # Normalize the image
            roi = roi_gray.astype('float') / 255.0
            
            # Reshape to match model's expected input
            # Add batch dimension and channel dimension
            roi = roi.reshape(1, 48, 48, 1)
            
            try:
                # Make prediction
                prediction = classifier.predict(roi)[0]
                
                # Get the label with highest probability
                label_index = prediction.argmax()
                label = emotion_labels[label_index]
                
                # Add confidence to the text
                confidence = prediction[label_index] * 100
                full_label = f"{label}: {confidence:.2f}%"
                
                # Put text on the frame
                label_position = (x, y-10)  # Slightly above the face
                cv2.putText(frame, full_label, label_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                
                # Optional: Print prediction details
                print(f"Predictions: {dict(zip(emotion_labels, prediction))}")
                
            except Exception as e:
                print(f"Prediction error: {e}")
                cv2.putText(frame, 'Prediction Error', (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        else:
            cv2.putText(frame, 'No Face Detected', (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detector', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()