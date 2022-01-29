import sys
import os
import numpy as np
from cv2 import cv2
from fer import FER
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders

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

# Global variables
sender_address = 'adityabhargav9876@gmail.com'
physical_address = 'XXX YYY ZZZ, India'
ocv_files = 'opencv-files'
cascade_classifier = 'haarcascade_frontalface_alt.xml'
subjects = ["", "Amber Heard", "Bill Gates", "Jason Momoa", "Paul Rudd", 
        "Scarlett Johansson"]
img_len = 500
img_width = 400

# Objects for face detection, recognition, and emotion detection
face_cascade = cv2.CascadeClassifier(os.path.join(ocv_files,
    cascade_classifier))
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
emotion_detector = FER()

def notify_authorities(image, sender_pass):
    receiver_address = 'adityabhargav9876@gmail.com'
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'EMERGENCY'
    content = 'Possible violent event at ' + physical_address 
    content = content + ". Possible attacker shown in image"
    message.attach(MIMEText(content, 'plain'))
    cv2.imwrite('WARNING.jpg', image)
    attach_file_name = 'WARNING.jpg'
    with open(attach_file_name, 'rb') as f:
        img_data = f.read()
    img = MIMEImage(img_data, name=attach_file_name)
    message.attach(img)
    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.starttls()
    session.login(sender_address, sender_pass)
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print("Notified authorities")

# Helper fn to draw rectangles on images
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
# Helper fn to draw text on images
def draw_text(img, text, x, y, emotion):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
    cv2.putText(img, emotion, (x, y+15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255,
        0), 2)

# Detects all faces in an image
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2,
            minNeighbors=5);   
    if (len(rects) == 0):
        return None, None   
    faces = []
    for rect in rects:
        (x, y, w, h) = rect
        faces.append(gray[y:y+w, x:x+h])
    return faces, rects

# Runs face and emotion recognition on images
def predict(test_img):
    img = test_img.copy()
    image2 = cv2.resize(img, (img_width, img_len))
    faces, rects = detect_face(image2)
    if faces is not None and rects is not None:
        for i in range(len(faces)):
            label, confidence = face_recognizer.predict(cv2.resize(faces[i],
                (100, 200)))
            (x, y, w, h) = rects[i]
            captured_emotions = emotion_detector.detect_emotions(image2[y:y+h,
                x:x+w])
            max_emotion = ''
            if (captured_emotions != []):
                max_emotion = max(captured_emotions[0]['emotions'], key =
                        captured_emotions[0]['emotions'].get)
                if (captured_emotions[0]['emotions'][max_emotion] < 0.5):
                    max_emotion = 'Emotion: Uncertain'
                else:
                    max_emotion = 'Emotion: ' + max_emotion
            label_text = subjects[label] + "(" + str(round(confidence)) + "%)"
            draw_rectangle(image2, rects[i])
            draw_text(image2, label_text, rects[i][0], rects[i][1]-5,
                    max_emotion)
        return image2
    else:
        draw_text(image2,"No face detected",6,25, "")
        return image2

# Capture video and run detection algorithms on captured frames
def video_cap():
    face_recognizer.read('eigen.cv2')
    video_cap = cv2.VideoCapture(0)
    while True:
        captured_image = video_cap.read()[1]
        cv2.imshow('Video', predict(captured_image))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Helper fn specifically to prepare training data
def prepare_training_data(data_folder_path):
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
            image2=cv2.resize(image, (img_width, img_len))
            cv2.imshow("Training on image...", image2)
            cv2.waitKey(100)
            pixels = detect_face(image2)[0]
            if pixels is not None:
                faces.append(cv2.resize(pixels[0], (100, 200)))
                labels.append(label)         
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()   
    return faces, labels

# Helper fn to predict all images in test dir
def predict_all():
    face_recognizer.read('eigen.cv2')
    testsubs = os.listdir("test-data")
    for imagename in testsubs:
        imagepath = "test-data/" + imagename
        test_img=cv2.imread(imagepath)
        predicted_img=predict(test_img)
        cv2.imshow(imagename, cv2.resize(predicted_img, (img_width, img_len)))
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        
# Helper fn to train classifier on all faces in training dir
def prepare_data():
    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write('eigen.cv2')
    print("Face recognizer written to disk")

def main():
    #prepare_data()
    #predict_all()
    #video_cap()
    return 0

if __name__ == "__main__":
    globals()[sys.argv[1]]()
