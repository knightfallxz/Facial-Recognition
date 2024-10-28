import os
import cv2
import numpy as np
import face_recognition
import pickle

def train_face_model(user):
    """
    Train the face recognition model using the images saved for the user.
    """
    images_path = os.path.join('registered_users', user)
    image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
    
    face_encodings = []

    for image_file in image_files:
        image_path = os.path.join(images_path, image_file)
        print(f"Processing image for user: {user}")
        
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Align and preprocess the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (500, 500))  # Resize to 500x500
        face_landmarks_list = face_recognition.face_landmarks(image)

        if face_landmarks_list:
            # Get the first face encoding
            face_encoding = face_recognition.face_encodings(image, known_face_locations=None)[0]
            face_encodings.append(face_encoding)
        else:
            print(f"No face landmarks found for {image_file}. Skipping this image.")

    if face_encodings:
        # Save the face encodings as a dictionary with the user's name
        model_path = 'models/face_recognition_model.pkl'
        model_data = {user: face_encodings}
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model trained and saved to '{model_path}'.")
    else:
        print(f"No valid face encodings found for user {user}. Model not saved.")

if __name__ == "__main__":
    user = input("Enter the username for training the face model: ")
    train_face_model(user)

