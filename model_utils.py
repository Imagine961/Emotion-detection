import cv2
import numpy as np

def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.reshape(1, 48, 48, 1) / 255.0
    return face
