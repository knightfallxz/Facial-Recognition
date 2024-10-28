import cv2
import numpy as np
import random
import face_recognition

def preprocess_and_save(image, save_path):
    """
    Apply advanced image processing techniques to enhance the input image.
    Includes sharpening, grayscale conversion, noise reduction, and resizing.
    Then saves the processed image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sharpen the image to enhance edges
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(gray, -1, kernel)

    # Resize to a standard size (e.g., 500x500)
    resized_image = cv2.resize(sharpened_image, (500, 500))

    # Apply Gaussian blur to reduce noise
    processed_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Save the processed image to the specified path
    cv2.imwrite(save_path, processed_image)

    print(f"Image saved at: {save_path}")

def preprocess_image(image):
    """
    Apply advanced image processing techniques to enhance the input image.
    This version does not save the processed image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sharpen the image to enhance edges
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(gray, -1, kernel)

    # Resize to a standard size (e.g., 500x500)
    resized_image = cv2.resize(sharpened_image, (500, 500))

    # Apply Gaussian blur to reduce noise
    processed_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    return processed_image

def augment_image(image):
    """
    Apply random image augmentations such as rotation and brightness adjustments.
    """
    # Randomly rotate the image by -10 to 10 degrees
    angle = random.uniform(-10, 10)
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    # Random brightness adjustment
    brightness_factor = random.uniform(0.9, 1.1)
    augmented_image = cv2.convertScaleAbs(rotated_image, alpha=brightness_factor, beta=0)

    return augmented_image

def align_face(image):
    """
    Align the face based on the position of the eyes to improve recognition accuracy.
    """
    face_landmarks_list = face_recognition.face_landmarks(image)
    if face_landmarks_list:
        landmarks = face_landmarks_list[0]
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']

        # Compute angle between eyes
        dx = right_eye[0][0] - left_eye[0][0]
        dy = right_eye[0][1] - left_eye[0][1]
        angle = np.degrees(np.arctan2(dy, dx)) - 180

        # Rotate the image to align the eyes
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        aligned_image = cv2.warpAffine(image, M, (w, h))

        return aligned_image
    else:
        print("No face landmarks found.")  # Debug print if no landmarks are found
    return image

