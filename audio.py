import soundfile
import numpy as np
import librosa
import glob
import os
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.metrics import accuracy_score
from itertools import permutations, combinations

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
        self.training_data = training_data
        if (os.path.exists("mlp_classifier.model")):
            print("Voice_Recognizer: Pre-trained model found...")
            self.model = pickle.load(open("mlp_classifier.model", 'rb'))
        else:
            print("Voice_Recognizer: No pre-trained model found...")
            self.model = self.train()
            
    def extract_feature(self, file_name, **kwargs):
        # Mel Frequency Cepstral Coefficient, represents the short-term power
        # spectrum of a sound
        mfcc = kwargs.get("mfcc")
        # the 12 different pitch classes
        chroma = kwargs.get("chroma")
        # Mel Spectrogram Frequency
        mel = kwargs.get("mel")
        # Each frame of a spectogram is divided into bands. for each band,
        # compute the contrast by comparing mean energy in top quartile to that
        # of the bottom quartile. high contrast = clear signals. 
        # low contrast = noise
        contrast = kwargs.get("contrast")
        # project chroma feature onto 6-dimensional basis
        tonnetz = kwargs.get("tonnetz")
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype = "float32")
            sample_rate = sound_file.samplerate
            if chroma or contrast:
                # short-time fourier transform
                # represents a signal in time-frequency domain by computing
                # discrete fourier transforms over short overlapping windows
                stft = np.abs(librosa.stft(X))
            # contains all the features stacked horizontally
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
            features = self.extract_feature(file, mfcc = True, chroma = True,
                    mel = True, contrast = True, tonnetz = True)
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
                True, mel = True, contrast = True, tonnetz = True)
        return self.model.predict(np.array([features]))[0]
        
    def research_extract_feature(self, file_name):
        print("extracting features for " + file_name)
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype = "float32")
            sample_rate = sound_file.samplerate
            stft = np.abs(librosa.stft(X))
            mfccs = np.mean(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 40).T, axis = 0)
            chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr = sample_rate).T, axis = 0)
            mel = np.mean(librosa.feature.melspectrogram(y = X, sr = sample_rate).T, axis = 0)
            contrast = np.mean(librosa.feature.spectral_contrast(S = stft, sr = sample_rate).T, axis = 0)
            tonnetz = np.mean(librosa.feature.tonnetz(y = librosa.effects.harmonic(X), sr = sample_rate).T, axis = 0)
        return mfccs, chroma, mel, contrast, tonnetz
        
    def research_load_data(self):
        mfccs_arr = []
        chromas_arr = []
        mels_arr = []
        contrasts_arr = []
        tonnetzs_arr = []
        y = []
        for file in glob.glob(os.path.join(self.training_data, "Actor_*", "*.wav")):
            basename = os.path.basename(file)
            emotion = self.int2emotion[basename.split("-")[2]]
            mfcc, chroma, mel, contrast, tonnetz = self.research_extract_feature(file)
            mfccs_arr.append(mfcc)
            chromas_arr.append(chroma)
            mels_arr.append(mel)
            contrasts_arr.append(contrast)
            tonnetzs_arr.append(tonnetz)
            y.append(emotion)
        pickle.dump(mfccs_arr, open("mfcc.list", "wb"))    
        pickle.dump(chromas_arr, open("chroma.list", "wb"))    
        pickle.dump(mels_arr, open("mel.list", "wb"))    
        pickle.dump(contrasts_arr, open("contrast.list", "wb"))    
        pickle.dump(tonnetzs_arr, open("tonnetz.list", "wb"))    
        pickle.dump(y, open("y.list", "wb"))
    
    def research_model(self):
        mfccs_arr = pickle.load(open("mfcc.list", "rb"))
        chromas_arr = pickle.load(open("chroma.list", "rb"))
        mels_arr = pickle.load(open("mel.list", "rb"))
        contrasts_arr = pickle.load(open("contrast.list", "rb"))
        tonnetzs_arr = pickle.load(open("tonnetz.list", "rb"))
        y = pickle.load(open("y.list", "rb"))
        X = []
        for i in range(len(mfccs_arr)):
            tmp = np.array([])
            tmp = np.hstack((tmp, mfccs_arr[i]))
            tmp = np.hstack((tmp, chromas_arr[i]))
            tmp = np.hstack((tmp, mels_arr[i]))
            tmp = np.hstack((tmp, contrasts_arr[i]))
            tmp = np.hstack((tmp, tonnetzs_arr[i]))
            X.append(tmp)
            del tmp
        X_train, X_test, y_train, y_test = train_test_split(np.array(X), y, test_size = 0.2, random_state = 7)
        print("[+] Number of training samples:", X_train.shape[0])
        print("[+] Number of testing samples:", X_test.shape[0])
        print("[+] Number of features:", X_train.shape[1])
        model_params = {
            'alpha': 0.01,
            'batch_size': 300,
            'epsilon': 0.01,
            'hidden_layer_sizes': (1000,), 
            'learning_rate': 'adaptive', 
            'max_iter': 5000,
            'solver': 'adam',
            'activation': 'tanh',
            'shuffle': True
        }
        model = MLPClassifier(**model_params)
        print("[*] Training the model...")
        model.fit(X_train, y_train)
        pickle.dump(model, open("mlp_classifier.model", "wb"))
        print("[*] Testing the model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
        print("Accuracy: {:.2f}%".format(accuracy*100))
        
class Recorder:

    def __init__(self):
        self.freq = 48000

    def record(self, duration = 10):
        print("recording...")
        recording = sd.rec(int(duration * self.freq), 
                samplerate = self.freq, channels = 1)
        sd.wait()
        print("writing recording to disk...")
        write("live_recording.wav", self.freq, recording)