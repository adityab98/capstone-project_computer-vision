import os
from audio import *
# from video import *

def main():
    """ocv_files = 'opencv_files'
    cascade_classifier = 'haarcascade_frontalface_alt.xml'
    subjects = ["", "Amber Heard", "Bill Gates", "Jason Momoa", "Paul Rudd", 
            "Scarlett Johansson"]
    img_len = 400
    img_width = 300
    face_rec = Face_Recognizers(os.path.join(ocv_files, cascade_classifier),
            img_len, img_width, subjects, os.path.join("data_files",
                "face_training"), os.path.join("data_files", "face_test"))
    face_rec.predict_all()
    face_rec.video_cap()"""
    voice_rec = Voice_Recognizer(os.path.join("data_files", "voice_training"))
    voice_rec.full_research()
    
if __name__ == "__main__":
    main()