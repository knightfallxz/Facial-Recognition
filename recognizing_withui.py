import cv2
import numpy as np
import pickle
from PyQt5.QtWidgets import QApplication, QMessageBox

class FaceRecognizer:
    def __init__(self):
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_frontalface_default.xml')
        self.labels = {}
        self.load_data()

    def load_data(self):
        try:
            with open('labels.pickle', 'rb') as f:
                og_labels = pickle.load(f)
                self.labels = {v: k for k, v in og_labels.items()}
            self.face_recognizer.read('./recognizer/training.yml')
        except FileNotFoundError:
            print("Error: 'labels.pickle' or 'training.yml' file not found. Please train the model first.")
            exit()

    def show_popup(self, message):
        app = QApplication([])
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Notification")
        msg.exec_()

    def recognize_faces(self):
        cam = cv2.VideoCapture(0)

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                id_, conf = self.face_recognizer.predict(roi_gray)

                if conf < 100:
                    name = self.labels.get(id_, "Unknown")
                    cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    if name != "Unknown":
                        self.show_popup("Facial recognition successful for "+ name)
                else:
                    cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Face Recognition', img)

            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.recognize_faces()
