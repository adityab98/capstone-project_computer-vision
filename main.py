from cv2 import cv2
import os
import numpy as np
from fer import FER

ocvfiles = 'opencv-files'
cclassifier = 'lbpcascade_frontalface.xml'
subjects = ["", "Amber Heard", "Bill Gates", "Jason Momoa", "Paul Rudd", 
        "Scarlett Johansson"]
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
img_len = 500
img_width = 400

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(os.path.join(ocvfiles, cclassifier))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2,
            minNeighbors=5);   
    if (len(faces) == 0):
        return None, None   
    pixels = []
    for face in faces:
        (x, y, w, h) = face
        pixels.append(gray[y:y+w, x:x+h])
    return pixels, faces

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

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()
    image2 = cv2.resize(img, (img_width, img_len))
    faces, rects = detect_face(image2)
    emotion_detector = FER(mtcnn=True)
    if faces is not None and rects is not None:
        for i in range(len(faces)):
            label, confidence = face_recognizer.predict(cv2.resize(faces[i],
                (100, 200)))
            (x, y, w, h) = rects[i]
            captured_emotions = emotion_detector.detect_emotions(image2[y:y+h,
                x:x+w])
            print(captured_emotions)
            label_text = subjects[label] + "(" + str(confidence) + "%)"
            draw_rectangle(image2, rects[i])
            draw_text(image2, label_text, rects[i][0], rects[i][1]-5)
        return image2
    else:
        draw_text(image2,"No face detected",6,25)
        return image2

def video_cap():
    face_recognizer.read('eigen.cv2')
    video_cap = cv2.VideoCapture(0)
    while True:
        captured_image = video_cap.read()[1]
        cv2.imshow('Video', predict(captured_image))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
    predict_all()
    #video_cap()

if __name__ == "__main__":
    main()
