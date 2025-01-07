import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('emotion_model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict emotion
def predict_emotion(face_roi):
    face_roi = cv2.resize(face_roi, (48, 48))  # Resize for the model
    face_roi = face_roi / 255.0  # Normalize pixel values
    face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension
    face_roi = np.expand_dims(face_roi, axis=0)   # Add batch dimension
    predictions = model.predict(face_roi)
    emotion_index = np.argmax(predictions)
    return emotion_labels[emotion_index]

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

print("Press 'q' to quit the application.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the camera.")
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_roi = gray_frame[y:y + h, x:x + w]

        # Predict the emotion for the detected face
        emotion = predict_emotion(face_roi)

        # Draw a rectangle around the face and display the emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame with annotations
    cv2.imshow('Real-Time Emotion Detector', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
