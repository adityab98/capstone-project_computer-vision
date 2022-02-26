import sys
import os
# from video import *
from audio import *
import numpy as np

def main():
    ocv_files = 'opencv_files'
    cascade_classifier = 'haarcascade_frontalface_alt.xml'
    subjects = ["", "Amber Heard", "Bill Gates", "Jason Momoa", "Paul Rudd", 
            "Scarlett Johansson"]
    img_len = 400
    img_width = 300
    # face_rec = Face_Recognizers(os.path.join(ocv_files, cascade_classifier),
            # img_len, img_width, subjects, os.path.join("data_files",
                # "face_training"), os.path.join("data_files", "face_test"))
    voice_rec = Voice_Recognizer(os.path.join("data_files", "voice_training"))
    # recorder = Recorder()
    # face_rec.predict_all()
    # face_rec.video_cap()
    # files = os.listdir("demo")
    # for file in files:
        # print(file, end = ' ')
        # print(voice_rec.predict(os.path.join("demo", file)))
    # recorder.record()
    # voice_rec.predict("live_recording.wav")
    voice_rec.research_model()
    # voice_rec.research_load_data()

    
if __name__ == "__main__":
    main()
