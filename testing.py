import tkinter as tk
import cv2
import numpy as np
from keras.models import model_from_json
from PIL import Image, ImageTk
import random
import csv

emotion_classes = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

class EmotionDetectorApp1:
    def __init__(self, window):
        self.window = window
        self.window.title("Movie Recommendation")

        # Camera feed on the left
        self.cap = cv2.VideoCapture(0)
        self.camera_label = tk.Label(window)
        self.camera_label.pack(side="left")
        self.update_camera_feed()

        # Display emotion on the right
        self.emotion_label = tk.Label(window, text="Emotion: ")
        self.emotion_label.pack(side="right")

        # Suggest button
        self.suggest_button = tk.Button(window, text="Suggest Movies", command=self.suggest_movies)
        self.suggest_button.pack()

        # Close button
        self.close_button = tk.Button(window, text="Close", command=self.close_app)
        self.close_button.pack()

        # Load the pre-trained model
        json_file = open(r"C:\Users\uditr\PycharmProjects\movie_recommendation_system_cnn\model\model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(r"C:\Users\uditr\PycharmProjects\movie_recommendation_system_cnn\model\new_model.h5")

        # CSV files for movie suggestions
        self.emotion_csv_files = {
            'Angry': r"C:\Users\uditr\Downloads\angry_movies (1).csv",
            'Disgust': r"C:\Users\uditr\Downloads\neutral.csv",
            'Fear': r"C:\Users\uditr\Downloads\fear (1).csv",
            'Happy': r"C:\Users\uditr\Downloads\happy.csv",
            'Neutral': r"C:\Users\uditr\Downloads\neutral.csv",
            'Sad': r"C:\Users\uditr\Downloads\sad.csv",
            'Surprise': r"C:\Users\uditr\Downloads\neutral.csv"
        }

    def update_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.camera_label.img = img
            self.camera_label.config(image=img)
        self.camera_label.after(10, self.update_camera_feed)

    def detect_emotion(self):
        ret, frame = self.cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        num_face = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        detected_emotion = "No Face Detected"

        for (x, y, w, h) in num_face:
            roi_gray_frame = gray_frame[y:y + h, x: x + w]
            cropped_img = np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1)
            if np.sum([roi_gray_frame]) != 0:
                emotion_prediction = self.model.predict(np.array([cropped_img]))
                max_index = int(np.argmax(emotion_prediction))
                detected_emotion = emotion_classes[max_index]
        return detected_emotion

    def suggest_movies(self):
        detected_emotion = self.detect_emotion()
        print(detected_emotion)
        csv_file = self.emotion_csv_files.get(detected_emotion)
        if csv_file:
            with open(csv_file, 'r', newline='', encoding='utf-8') as input_csv:
                csv_reader = csv.reader(input_csv)
                movie_titles = [row[0] for row in csv_reader]
                random_movie_titles = random.sample(movie_titles, 5)
                self.emotion_label.config(text=f"Emotion: {detected_emotion}\n\n\nSuggested Movies:\n\n-" + "\n-".join(random_movie_titles))
        else:
            self.emotion_label.config(text="No movie suggestions for the detected emotion")

    def close_app(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.window.quit()

if __name__ == '__main__':
    root = tk.Tk()
    app = EmotionDetectorApp1(root)
    root.mainloop()
