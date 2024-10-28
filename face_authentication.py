import cv2
import os
import pickle
import face_recognition
import sys
from utils import preprocess_image, align_face

def load_model(model_path):
    """
    Load the trained face recognition model from the specified path.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def capture_live_image():
    """
    Capture a live image using the webcam.
    """
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    if ret:
        cv2.imshow('Authenticate Face', frame)
        cv2.waitKey(1)
    video_capture.release()
    cv2.destroyAllWindows()
    return frame

def authenticate_face(username):
    """
    Authenticate the user's face by comparing a live captured image
    against the registered user's face encodings.
    """
    model_path = 'models/face_recognition_model.pkl'
    
    if not os.path.exists(model_path):
        print("Model not found. Please register at least one user.")
        return False

    # Load the trained face recognition model
    face_recognition_model = load_model(model_path)

    # Check if the model is a dictionary
    if not isinstance(face_recognition_model, dict):
        print("Loaded model is not in the expected format.")
        return False

    # Check if the username exists in the model
    if username not in face_recognition_model:
        print(f"No registered face encoding found for {username}.")
        return False

    # Capture live image for authentication
    print("Looking for a registered face. Please look at the camera.")
    live_image = capture_live_image()

    # Extract face encoding from the live image
    live_image_encoding = face_recognition.face_encodings(live_image)
    if len(live_image_encoding) == 0:
        print("No face landmarks found in the live image. Please try again.")
        return False

    # Compare the live image encoding with the registered encodings
    registered_encodings = face_recognition_model[username]
    results = face_recognition.compare_faces(registered_encodings, live_image_encoding[0])

    if any(results):  # Check if any encoding matches
        print(f"Authentication successful. Welcome, {username}!")
        sys.exit()
        return True
    else:
        print("Authentication failed. Face does not match.")
        return False

