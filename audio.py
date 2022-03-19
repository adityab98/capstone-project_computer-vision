import soundfile
import numpy as np
import librosa
import glob
import os
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
    
    def predict_ensemble(self, audio_src_file):
        mfcc, chroma, mel, contrast, tonnetz = self.extract_feature(audio_src_file)
        X = np.array([])
        X = np.hstack((X, mfcc))
        X = np.hstack((X, chroma))
        X = np.hstack((X, mel))
        X = np.hstack((X, contrast))
        X = np.hstack((X, tonnetz))
        X = [X]
        X = np.array(X)
        adam_model = pickle.load(open("mlp_adam.model", "rb"))
        lbgfs_model = pickle.load(open("mlp_lbgfs.model", "rb"))
        ensemble_model = pickle.load(open("ensembled.model", "rb"))
        adam_preds = adam_model.predict_proba(X)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        lbgfs_preds = lbgfs_model.predict_proba(X)
        stacked_x = adam_preds
        stacked_x = dstack((stacked_x, lbgfs_preds))
        stacked_x = stacked_x.reshape((stacked_x.shape[0], stacked_x.shape[1]*stacked_x.shape[2]))
        return ensemble_model.predict(stacked_x)[0]
    
    def extract_feature(self, file_name):
        # print("extracting features for " + file_name)
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
    
    def save_training_data(self):
        mfccs_arr = []
        chromas_arr = []
        mels_arr = []
        contrasts_arr = []
        tonnetzs_arr = []
        y = []
        for file in glob.glob(os.path.join(self.training_data, "Actor_*", "*.wav")):
            basename = os.path.basename(file)
            emotion = self.int2emotion[basename.split("-")[2]]
            mfcc, chroma, mel, contrast, tonnetz = self.extract_feature(file)
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
        
    def full_research(self):
        print("[*] Loading saved features...")
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
            # tmp = np.hstack((tmp, contrasts_arr[i]))
            # tmp = np.hstack((tmp, tonnetzs_arr[i]))
            X.append(tmp)
            del tmp
        print("[*] Preprocessing data...")
        X = np.array(X)
        scaler = MinMaxScaler(feature_range=(-1,1))
        X = scaler.fit_transform(X)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X_train_l0, X_test_l0, y_train_l0, y_test_l0 = train_test_split(X, y, test_size = 0.2)
        print("[*] Training MLPClassifier NN with adam solver...")
        adam_model_params = {
            'alpha': 0.0001,
            'batch_size': 300,
            'epsilon': 0.01,
            'hidden_layer_sizes': (1000,),
            'max_iter': 5000,
            'solver': 'adam',
            'activation': 'tanh',
            'shuffle': True,
            'n_iter_no_change': 50,
        }
        adam_model = MLPClassifier(**adam_model_params)
        adam_model.fit(X_train_l0, y_train_l0)
        print("[*] Testing MLPClassifier NN with adam solver...")
        accuracy = accuracy_score(y_true = y_test_l0, y_pred = adam_model.predict(X_test_l0))
        print("\tAccuracy: {:.2f}%".format(accuracy*100))
        print("[*] Training MLPClassifier NN with lbfgs solver...")
        lbgfs_model_params = {
            'alpha': 0.0001,
            'batch_size': 300,
            'hidden_layer_sizes': (1000,), 
            'max_iter': 5000,
            'solver': 'lbfgs',
            'activation': 'tanh',
        }
        lbgfs_model = MLPClassifier(**lbgfs_model_params)
        lbgfs_model.fit(X_train_l0, y_train_l0)
        print("[*] Testing MLPClassifier NN with lbfgs solver...")
        accuracy = accuracy_score(y_true = y_test_l0, y_pred = lbgfs_model.predict(X_test_l0))
        print("\tAccuracy: {:.2f}%".format(accuracy*100))
        X_train_l1, X_test_l1, y_train_l1, y_test_l1 = train_test_split(np.array(X_test_l0), y_test_l0, test_size = 0.2)
        adam_preds = adam_model.predict_proba(X_train_l1)
        lbgfs_preds = lbgfs_model.predict_proba(X_train_l1)
        stackX = adam_preds
        stackX = np.dstack((stackX, lbgfs_preds))
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
        model = LogisticRegression()
        print("[*] Training LogisticRegression ML ensemble model...")
        model.fit(stackX, y_train_l1)
        adam_preds = adam_model.predict_proba(X_test_l1)
        lbgfs_preds = lbgfs_model.predict_proba(X_test_l1)
        stackX = adam_preds
        stackX = np.dstack((stackX, lbgfs_preds))
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
        print("[*] Testing LogisticRegression ML ensemble model...")
        accuracy = accuracy_score(y_true = y_test_l1, y_pred = model.predict(stackX))
        print("\tAccuracy: {:.2f}%".format(accuracy*100))