import cv2
import os

def take_photos(user_name, num_photos=20):
    # Ensure images folder exists
    images_path = f"./images/{user_name}"
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_frontalface_default.xml')

    # Start capturing from webcam
    cap = cv2.VideoCapture(0)
    photo_count = 0

    while photo_count < num_photos:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Save the face region
            face_roi = gray[y:y+h, x:x+w]
            img_path = os.path.join(images_path, f"photo_{photo_count}.jpg")
            cv2.imwrite(img_path, face_roi)
            photo_count += 1

            print(f"Photo {photo_count} taken and saved to {img_path}")

        # Show the frame
        cv2.imshow('Capturing Photos', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_name = input("Enter your name: ").lower().replace(" ", "_")
    take_photos(user_name)
