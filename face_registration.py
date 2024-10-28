import os
import cv2
import face_recognition
import sys

def register_face(user):
    """
    Register a new user's face by capturing multiple images and saving them.
    """
    user_dir = os.path.join('registered_users', user)
    os.makedirs(user_dir, exist_ok=True)

    print("Please look at the camera and slightly move your head in different angles.")
    
    video_capture = cv2.VideoCapture(0)  # Open the default camera
    captured_images = 0
    max_images = 20

    while captured_images < max_images:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image. Exiting...")
            break

        # Process the image for face encoding
        face_landmarks_list = face_recognition.face_landmarks(frame)
        if face_landmarks_list:
            # Save the captured image if landmarks are found
            image_path = os.path.join(user_dir, f"{user}_{captured_images}.jpg")
            cv2.imwrite(image_path, frame)
            captured_images += 1
            print(f"Image saved at: {image_path}")
            print(f"Captured and processed {captured_images}/{max_images} images with face encoding.")
        else:
            print("No face landmarks found. Retrying...")

    video_capture.release()
    cv2.destroyAllWindows()
    print(f"Face registration completed for user {user}.")

