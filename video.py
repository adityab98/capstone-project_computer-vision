import os
from cv2 import cv2
from fer import FER
import numpy as np

class Image_Helper:
    # Helper fn to draw rectangles on images
    @staticmethod
    def draw_rectangle(img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # Helper fn to draw text on images
    @staticmethod
    def draw_text(img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0),
                2)

    @staticmethod
    def video_cap():
        video_cap = cv2.VideoCapture(0)
        captured_image = video_cap.read()[1]
        return captured_image

class Face_Recognizers:

    def __init__(self, cascade_classifier, img_len, img_width, subjects,
            training_data, test_data):
        self.cascade_classifier = cascade_classifier
        self.face_cascade = cv2.CascadeClassifier(self.cascade_classifier)
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.subjects = subjects
        self.img_len = img_len
        self.img_width = img_width
        self.training_data = training_data
        self.test_data = test_data
        self.emo_recognizer = Emotion_Recognizer()
        if (os.path.exists('face_recognizer.cv2')):
            print("Face_Recognizer: Pre-trained model found...")
            self.face_recognizer.read('face_recognizer.cv2')
        else:
            print("""Face_Recognizer: No pre-trained model found. Training
                    model...""")
            self.train()
    
    def predict_all(self):
        test_subjects = os.listdir(self.test_data)
        for image_name in test_subjects:
            image_path = os.path.join(self.test_data, image_name)
            print("predicting..." + image_path)
            test_img = cv2.imread(image_path)
            predicted_img = self.predict(test_img)
            cv2.imshow(image_name, cv2.resize(predicted_img, (self.img_width,
                self.img_len)))
            cv2.waitKey(3000)
            cv2.destroyAllWindows()

    def prepare_training_data(self, data_folder_path):
        dirs = os.listdir(data_folder_path)
        faces = []
        labels = []
        for dir_name in dirs:
            if not dir_name.startswith("s"):
                continue;      
            label = int(dir_name.replace("s", ""))
            subject_dir_path = data_folder_path + "/" + dir_name
            subject_images_names = os.listdir(subject_dir_path)
            for image_name in subject_images_names:
                if image_name.startswith("."):
                    continue;
                image_path = subject_dir_path + "/" + image_name
                image = cv2.imread(image_path)
                image2=cv2.resize(image, (self.img_width, self.img_len))
                cv2.imshow("Training on image...", image2)
                cv2.waitKey(100)
                pixels = self.detect_face(image2)[0]
                if pixels is not None:
                    faces.append(cv2.resize(pixels[0], (100, 200)))
                    labels.append(label)         
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()   
        return faces, labels

    def train(self):
        print("Preparing data...")
        faces, labels = self.prepare_training_data(self.training_data)
        print("Data prepared")
        print("Total faces: ", len(faces))
        print("Total labels: ", len(labels))
        self.face_recognizer.train(faces, np.array(labels))
        self.face_recognizer.write('face_recognizer.cv2')
        print("Face recognizer written to disk")

    # Detects all faces in an image
    def detect_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2,
                minNeighbors=5);   
        if (len(rects) == 0):
            return None, None   
        faces = []
        for rect in rects:
            (x, y, w, h) = rect
            faces.append(gray[y:y+w, x:x+h])
        return faces, rects

    def predict(self, test_img):
        img = test_img.copy()
        img = cv2.resize(img, (self.img_width, self.img_len))
        faces, rects = self.detect_face(img)
        predictions = []
        if faces is not None and rects is not None:
            for i in range(len(faces)):
                face = cv2.resize(faces[i], (100, 200))
                label, confidence = self.face_recognizer.predict(face)
                confidence = round(confidence)
                emotion = self.emo_recognizer.predict(img, rects[i])
                if (confidence < 50):
                    predictions.append(("Unknown", confidence, emotion))
                else:
                    predictions.append((label, confidence, emotion))
                label_text = self.subjects[label] + "(" + str(confidence) + "%)"
                Image_Helper.draw_rectangle(img, rects[i])
                Image_Helper.draw_text(img, label_text, rects[i][0],
                        rects[i][1]-5)
                Image_Helper.draw_text(img, emotion, rects[i][0],
                        rects[i][1]+10)
            return img
        else:
            Image_Helper.draw_text(img, "No face detected", 6, 25)
            return img

    def video_cap(self):
        video_capture = cv2.VideoCapture(0)
        while True:
            rect, frames = video_capture.read()
            cv2.imshow("Live", cv2.resize(self.predict(frames), (self.img_width,
                self.img_len)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

class Emotion_Recognizer:

    def __init__(self):
        self.emotion_detector = FER()

    def predict(self, img, rect):
        (x, y, w, h) = rect
        captured_emotions = self.emotion_detector.detect_emotions(img[y:y+h,
            x:x+w])
        max_emotion = ''
        if (captured_emotions != []):
            max_emotion = max(captured_emotions[0]['emotions'], key =
                    captured_emotions[0]['emotions'].get)
            if (captured_emotions[0]['emotions'][max_emotion] < 0.5):
                max_emotion = 'Emotion: Uncertain'
            else:
                max_emotion = 'Emotion: ' + max_emotion
        return max_emotion

    # tensorflow config
    #Assume that the number of cores per socket in the machine is denoted as
    #NUM_PARALLEL_EXEC_UNITS
    #  when NUM_PARALLEL_EXEC_UNITS=0 the system chooses appropriate settings 
    #import tensorflow as tf
    #config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0, 
    #                        inter_op_parallelism_threads=0, 
    #                        allow_soft_placement=True,
    #                        device_count = {'CPU': 0})
    #session = tf.compat.v1.Session(config=config)
