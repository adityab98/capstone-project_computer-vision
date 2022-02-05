#!/bin/python

import sys
from video import *
from notifier import *
from audio import *

def main(password):
    sender_address = 'adityabhargav9876@gmail.com'
    physical_address = 'XXX YYY ZZZ, India'
    ocv_files = 'opencv_files'
    cascade_classifier = 'haarcascade_frontalface_alt.xml'
    subjects = ["", "Amber Heard", "Bill Gates", "Jason Momoa", "Paul Rudd", 
            "Scarlett Johansson"]
    negative_emotions = ["sad", "angry", "fearful", "disgust", "surprised"]
    img_len = 500
    img_width = 400
    face_rec = Face_Recognizers(os.path.join(ocv_files, cascade_classifier),
            img_len, img_width, subjects, os.path.join("data_files",
                "face_training"), os.path.join("data_files", "face_test"))
    notifier = Notifier(sender_address, physical_address, sender_address,
            password)
    voice_rec = Voice_Recognizer(os.path.join("data_files", "voice_training"))
    recorder = Recorder()

    while True:
        recorder.record()
        voice_emo = voice_rec.predict("live_recording.wav")
        if (voice_emo in negative_emotions):
            img = Image_Helper.video_cap()
            predictions = face_rec.predict(img)
            unknown = False
            for prediction in predictions:
                if (prediction[0] == "Unknown" and prediction[2] in
                        negative_emotions):
                    notifier.notify(img)
                    unknown = True
                    #keep scanning for more information
            #if known then **DO STUFF**

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Please enter your gmail password as an argument to the program.")
    main(sys.argv[1])
