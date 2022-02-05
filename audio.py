import soundfile # to read audio file
import numpy as np
import librosa # to extract speech features
import glob
import os
import pickle
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
import sounddevice as sd
from scipy.io.wavfile import write

class Voice_Recognizer:

    def __init__(self, training_data):
        self.int2emotion = {
                "01": "neutral", 
                "02": "calm", 
                "03": "happy", 
                "04": "sad", 
                "05": "angry", 
                "06": "fearful", 
                "07": "disgust", 
                "08": "surprised"
        }
        self.available_emotions = {
                "angry", 
                "sad", 
                "fearful", 
                "disgust",
                "surprise"
        }
        self.training_data = training_data
        if (os.path.exists("mlp_classifier.model")):
            print("Voice_Recognizer: Pre-trained model found...")
            self.model = pickle.load(open("mlp_classifier.model", 'rb'))
        else:
            print("Voice_Recognizer: No pre-trained model found...")
            self.model = self.train()
            
    def extract_feature(self, file_name, **kwargs):
        mfcc = kwargs.get("mfcc")
        chroma = kwargs.get("chroma")
        mel = kwargs.get("mel")
        contrast = kwargs.get("contrast")
        tonnetz = kwargs.get("tonnetz")
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype = "float32")
            sample_rate = sound_file.samplerate
            if chroma or contrast:
                stft = np.abs(librosa.stft(X))
            result = np.array([])
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y = X, sr = sample_rate,
                    n_mfcc = 40).T, axis = 0)
                result = np.hstack((result, mfccs))
            if chroma:
                chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr =
                    sample_rate).T, axis = 0)
                result = np.hstack((result, chroma))
            if mel:
                mel = np.mean(librosa.feature.melspectrogram(X, sr =
                    sample_rate).T, axis = 0)
                result = np.hstack((result, mel))
            if contrast:
                contrast = np.mean(librosa.feature.spectral_contrast(S = stft,
                    sr = sample_rate).T, axis = 0)
                result = np.hstack((result, contrast))
            if tonnetz:
                tonnetz = np.mean(librosa.feature.tonnetz(y =
                    librosa.effects.harmonic(X), sr = sample_rate).T, axis = 0)
                result = np.hstack((result, tonnetz))
        return result

    def load_data(self):
        X, y = [], []
        for file in glob.glob(os.path.join(self.training_data, "Actor_*",
            "*.wav")):
            basename = os.path.basename(file)
            emotion = self.int2emotion[basename.split("-")[2]]
            if emotion not in self.available_emotions:
                continue
            features = self.extract_feature(file, mfcc = True, chroma = True,
                    mel = True)
            X.append(features)
            y.append(emotion)
        return np.array(X), y
    
    def train(self):
        X_train, y_train = self.load_data()
        print("[+] Number of training samples:", X_train.shape[0])
        print("[+] Number of features:", X_train.shape[1])
        model_params = {
            'alpha': 0.01,
            'batch_size': 256,
            'epsilon': 1e-08, 
            'hidden_layer_sizes': (300,), 
            'learning_rate': 'adaptive', 
            'max_iter': 500, 
        }
        model = MLPClassifier(**model_params)
        print("[*] Training the model...")
        model.fit(X_train, y_train)
        pickle.dump(model, open("mlp_classifier.model", "wb"))
        return model

    def predict(self, audio_src_file):
        features = self.extract_feature(audio_src_file, mfcc = True, chroma =
                True, mel = True)
        return self.model.predict(np.array([features]))[0]

class Recorder:

    def __init__(self):
        self.freq = 48000
        self.duration = 10

    def record(self):
        recording = sd.rec(int(self.duration * self.freq), 
                samplerate = self.freq, channels = 2)
        sd.wait()
        write("live_recording.wav", self.freq, recording)
