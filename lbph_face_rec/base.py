from cv2 import cv2
import os
import numpy as np

#locn of files required by opencv
ocvfiles = 'opencv-files'
#using lbp as its faster on embedded devices
cclassifier = 'lbpcascade_frontalface.xml'
subjects = ["","Amber Heard","Bill Gates","Jason Momoa","Paul Rudd","Scarlett Johansson"]
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(os.path.join(ocvfiles, cclassifier))
    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);   
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None   
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0] 
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

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
            image2=cv2.resize(image,(400,500))
            cv2.imshow("Training on image...", image2)
            cv2.waitKey(100)
            face, rect = detect_face(image2)
            if face is not None:
                faces.append(cv2.resize(face,(100,200)))
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
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    image2=cv2.resize(img,(400,500))
    face, rect = detect_face(image2)
    #predict the image using our face recognizer 
    if face is not None:
        label, confidence = face_recognizer.predict(cv2.resize(face,(100,200)))
        #get name of respective label returned by face recognizer
        label_text = subjects[label]
        
        #draw a rectangle around face detected
        draw_rectangle(image2, rect)
        #draw name of predicted person
        draw_text(image2, label_text, rect[0], rect[1]-5)
        
        return image2
    else:
        draw_text(image2,"No face detected",6,25)
        return image2

def predict_all():
    testsubs = os.listdir("test-data")
    for imagename in testsubs:
        imagepath = "test-data/" + imagename
        test_img=cv2.imread(imagepath)
        predicted_img=predict(test_img)
        cv2.imshow(imagename,cv2.resize(predicted_img,(400,500)))
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        
def videoCap():
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:   
        check, frame = webcam.read()
        print(check)
        print(frame)
        frame=cv2.flip(frame,1)
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            #os.chdir(r'C:\Users\admin\Desktop\facerec\test-data')
            #fname='webcamtest'+str(random.randint(100,200))+'.jpg'
            #cv2.imwrite(filename=fname, img=frame)
            webcam.release()
            cv2.imshow('Captured image',frame)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            #os.chdir(r'C:\Users\admin\Desktop\facerec')
            #print("Image saved!")
            #image=cv2.imread("test-data/"+fname)
            img=predict(frame)
            cv2.imshow('Captured image prediction',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

def main():
    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    face_recognizer.train(faces, np.array(labels))
    #face_recognizer.write('eigen.cv2')
    #print("Face recognizer written to disk")
    #face_recognizer.read('eigen.cv2')
    predict_all()

if __name__ == "__main__":
    main()
