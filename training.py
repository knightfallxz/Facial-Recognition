import os
import cv2
import numpy as np
from PIL import Image
import pickle

class Trainer:
    def __init__(self):
        self.cascade_classifier = cv2.CascadeClassifier('./classifiers/haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.img_path = os.path.join(self.base_dir, 'images')
        self.label_ids = {}
        self.x_train = []
        self.y_train = []

    def train_faces(self):
        current_id = 0
        for path, dirnames, files in os.walk(self.img_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    file_path = os.path.join(path, file)
                    label = os.path.basename(path).replace(" ", "-").lower()
                    if label not in self.label_ids:
                        self.label_ids[label] = current_id
                        current_id += 1
                    id_ = self.label_ids[label]
                    pil_image = Image.open(file_path).convert("L")
                    size = (500, 500)
                    final_image = pil_image.resize(size, Image.Resampling.LANCZOS)
                    image_array = np.array(final_image, "uint8")
                    faces = self.cascade_classifier.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
                    for (x, y, w, h) in faces:
                        roi = image_array[y:y+h, x:x+w]
                        self.x_train.append(roi)
                        self.y_train.append(id_)

        with open("labels.pickle", "wb") as f:
            pickle.dump(self.label_ids, f)
        self.recognizer.train(self.x_train, np.array(self.y_train))
        self.recognizer.save("recognizer/training.yml")
        print("Training complete. Model saved to 'recognizer/training.yml'.")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_faces()
