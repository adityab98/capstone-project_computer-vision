#!/bin/python

import sys
import os
from video import *
from notifier import *
from audio import *

def main():
    if (len(sys.argv) != 2):
        print("No valid password specified. Setting password as None.")
        password = None
    else:
        password = sys.argv[1]
    sender_address = 'adityabhargav9876@gmail.com'
    physical_address = 'XXX YYY ZZZ, India'
    ocv_files = 'opencv_files'
    cascade_classifier = 'haarcascade_frontalface_alt.xml'
    subjects = ["", "Amber Heard", "Bill Gates", "Jason Momoa", "Paul Rudd", 
            "Scarlett Johansson"]
    img_len = 400
    img_width = 300
    face_rec = Face_Recognizers(os.path.join(ocv_files, cascade_classifier),
            img_len, img_width, subjects, os.path.join("data_files",
                "face_training"), os.path.join("data_files", "face_test"))
    notifier = Notifier(sender_address, physical_address, sender_address,
            password)
    recorder = Recorder()
    voice_rec = Voice_Recognizer(os.path.join("data_files", "voice_training"))
    files = os.listdir("demo")
    for file in files:
        print(file, end = ' ')
        print(voice_rec.predict(os.path.join("demo", file)))
    return 0

if __name__ == "__main__":
    main()