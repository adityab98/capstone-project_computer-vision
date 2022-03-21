import soundfile
import numpy as np
import librosa
import glob
import os
import pickle
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

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
        self.tess2emotion = {
            "angry": "angry",
            "fear": "fearful",
            "happy": "happy",
            "disgust": "disgust",
            "neutral": "neutral",
            "ps": "surprised",
            "sad": "sad"
        }
        self.training_data = training_data
    def save_ravdess_features(self):
        mfccs_arr = []
        chromas_arr = []
        mels_arr = []
        contrasts_arr = []
        tonnetzs_arr = []
        y = []
        for file in glob.glob(os.path.join(self.training_data, "ravdess", "*.wav")):
            basename = os.path.basename(file)
            emotion = self.int2emotion[basename.split("-")[2]]
            mfcc, chroma, mel, contrast, tonnetz = self.extract_feature(file)
            mfccs_arr.append(mfcc)
            chromas_arr.append(chroma)
            mels_arr.append(mel)
            contrasts_arr.append(contrast)
            tonnetzs_arr.append(tonnetz)
            y.append(emotion)
        pickle.dump(mfccs_arr, open("ravdess_mfcc.list", "wb"))    
        pickle.dump(chromas_arr, open("ravdess_chroma.list", "wb"))    
        pickle.dump(mels_arr, open("ravdess_mel.list", "wb"))    
        pickle.dump(contrasts_arr, open("ravdess_contrast.list", "wb"))    
        pickle.dump(tonnetzs_arr, open("ravdess_tonnetz.list", "wb"))    
        pickle.dump(y, open("ravdess_y.list", "wb"))     
    def save_tess_features(self):
        mfccs_arr = []
        chromas_arr = []
        mels_arr = []
        contrasts_arr = []
        tonnetzs_arr = []
        y = []
        for file in glob.glob(os.path.join(self.training_data, "tess", "*.wav")):
            basename = os.path.basename(file)
            emotion = self.tess2emotion[basename.split("_")[2].split('.')[0]]
            mfcc, chroma, mel, contrast, tonnetz = self.extract_feature(file)
            mfccs_arr.append(mfcc)
            chromas_arr.append(chroma)
            mels_arr.append(mel)
            contrasts_arr.append(contrast)
            tonnetzs_arr.append(tonnetz)
            y.append(emotion)
        pickle.dump(mfccs_arr, open("tess_mfcc.list", "wb"))    
        pickle.dump(chromas_arr, open("tess_chroma.list", "wb"))    
        pickle.dump(mels_arr, open("tess_mel.list", "wb"))    
        pickle.dump(contrasts_arr, open("tess_contrast.list", "wb"))    
        pickle.dump(tonnetzs_arr, open("tess_tonnetz.list", "wb"))    
        pickle.dump(y, open("tess_y.list", "wb"))
    
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
    
    def normalize(self, X):
        X = np.array(X)
        scaler = MinMaxScaler(feature_range=(-1,1))
        X = scaler.fit_transform(X)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        return X
    
    def get_stacked(self, X, models):
        stackX = None
        for model in models:
            probs = model.predict_proba(X)
            if stackX is None:
                stackX = probs
            else: stackX = np.dstack((stackX, probs))
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
        return stackX
    
    def full_research(self):
        print("[*]L0 Loading saved RAVDESS features...")
        mfccs_arr = pickle.load(open("ravdess_mfcc.list", "rb"))
        chromas_arr = pickle.load(open("ravdess_chroma.list", "rb"))
        mels_arr = pickle.load(open("ravdess_mel.list", "rb"))
        contrasts_arr = pickle.load(open("ravdess_contrast.list", "rb"))
        tonnetzs_arr = pickle.load(open("ravdess_tonnetz.list", "rb"))
        y = pickle.load(open("ravdess_y.list", "rb"))
        print("[*]L0 Loading saved TESS features...")
        mfccs_arr.extend(pickle.load(open("tess_mfcc.list", "rb")))
        chromas_arr.extend(pickle.load(open("tess_chroma.list", "rb")))
        mels_arr.extend(pickle.load(open("tess_mel.list", "rb")))
        contrasts_arr.extend(pickle.load(open("tess_contrast.list", "rb")))
        tonnetzs_arr.extend(pickle.load(open("tess_tonnetz.list", "rb")))
        y.extend(pickle.load(open("tess_y.list", "rb")))
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
        print("[*]L0 Preprocessing data...")
        X = self.normalize(X)
        X_train_l0, X_test_l0, y_train_l0, y_test_l0 = train_test_split(X, y, test_size = 0.4, random_state = 12)
        # print("[*]L0 Training MLPClassifier NN with adam solver...")
        # adam_model_params = {'batch_size': 300, 'epsilon': 0.01, 'hidden_layer_sizes': (1000,), 'max_iter': 5000, 'solver': 'adam', 'activation': 'tanh', 'n_iter_no_change': 50,}
        # adam_model = MLPClassifier(**adam_model_params)
        # adam_model.fit(X_train_l0, y_train_l0)
        # print("[*]L0 Testing MLPClassifier NN with adam solver...")
        # accuracy = accuracy_score(y_true = y_test_l0, y_pred = adam_model.predict(X_test_l0))
        # print("\tAccuracy: {:.2f}%".format(accuracy*100))
        # print("[*]L0 Training MLPClassifier NN with lbfgs solver...")
        # lbgfs_model_params = {'batch_size': 300, 'hidden_layer_sizes': (1000,),  'max_iter': 5000, 'solver': 'lbfgs', 'activation': 'tanh'}
        # lbgfs_model = MLPClassifier(**lbgfs_model_params)
        # lbgfs_model.fit(X_train_l0, y_train_l0)
        # print("[*]L0 Testing MLPClassifier NN with lbfgs solver...")
        # accuracy = accuracy_score(y_true = y_test_l0, y_pred = lbgfs_model.predict(X_test_l0))
        # print("\tAccuracy: {:.2f}%".format(accuracy*100))
        adam_model = pickle.load(open("adam.model", "rb"))###############
        lbgfs_model = pickle.load(open("lbgfs.model", "rb"))#############
        X_train_l1, X_test_l1, y_train_l1, y_test_l1 = train_test_split(X_test_l0, y_test_l0, test_size = 0.2)
        stackX = self.get_stacked(X_train_l1, [adam_model, lbgfs_model])
        model = LogisticRegression()
        print("[*]L1 Training LogisticRegression ML ensemble model...")
        model.fit(stackX, y_train_l1)
        stackX = self.get_stacked(X_test_l1, [adam_model, lbgfs_model])
        print("[*]L1 Testing LogisticRegression ML ensemble model...")
        accuracy = accuracy_score(y_true = y_test_l1, y_pred = model.predict(stackX))
        print("\tAccuracy: {:.2f}%".format(accuracy*100))
        # print("[*] Saving models to disk...")
        # pickle.dump(model, open("ensemble.model", "wb"))
        # pickle.dump(adam_model, open("adam.model", "wb"))
        # pickle.dump(lbgfs_model, open("lbgfs.model", "wb"))
        
    def finalize_models(self):
        print("[*]L0 Loading saved RAVDESS features...")
        mfccs_arr = pickle.load(open("ravdess_mfcc.list", "rb"))
        chromas_arr = pickle.load(open("ravdess_chroma.list", "rb"))
        mels_arr = pickle.load(open("ravdess_mel.list", "rb"))
        contrasts_arr = pickle.load(open("ravdess_contrast.list", "rb"))
        tonnetzs_arr = pickle.load(open("ravdess_tonnetz.list", "rb"))
        y = pickle.load(open("ravdess_y.list", "rb"))
        print("[*]L0 Loading saved TESS features...")
        mfccs_arr.extend(pickle.load(open("tess_mfcc.list", "rb")))
        chromas_arr.extend(pickle.load(open("tess_chroma.list", "rb")))
        mels_arr.extend(pickle.load(open("tess_mel.list", "rb")))
        contrasts_arr.extend(pickle.load(open("tess_contrast.list", "rb")))
        tonnetzs_arr.extend(pickle.load(open("tess_tonnetz.list", "rb")))
        y.extend(pickle.load(open("tess_y.list", "rb")))
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
        print("[*]L0 Preprocessing data...")
        X = self.normalize(X)
        X_train_l0, X_train_l1, y_train_l0, y_train_l1 = train_test_split(X, y, test_size = 0.4)
        print("[*]L0 Training MLPClassifier NN with adam solver...")
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
        print("[*]L0 Training MLPClassifier NN with lbfgs solver...")
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
        stackX = self.get_stacked(X_train_l1, [adam_model, lbgfs_model])
        model = LogisticRegression()
        print("[*]L1 Training LogisticRegression ML ensemble model...")
        model.fit(stackX, y_train_l1)
        print("[*] Saving models to disk...")
        pickle.dump(model, open("ensemble.model", "wb"))
        pickle.dump(adam_model, open("adam.model", "wb"))
        pickle.dump(lbgfs_model, open("lbgfs.model", "wb"))        
        
    def predict(self, audio_src_file):
        mfcc, chroma, mel, contrast, tonnetz = self.extract_feature(audio_src_file)
        X = np.array([])
        X = np.hstack((X, mfcc))
        X = np.hstack((X, chroma))
        X = np.hstack((X, mel))
        X = np.hstack((X, contrast))
        X = np.hstack((X, tonnetz))
        X = [X]
        X = self.normalize(X)
        adam_model = pickle.load(open("adam.model", "rb"))
        lbgfs_model = pickle.load(open("lbgfs.model", "rb"))
        ensemble_model = pickle.load(open("ensemble.model", "rb"))
        stackX = self.get_stacked(X, [adam_model, lbgfs_model])
        return ensemble_model.predict(stackX)[0]

def main():
    voice_rec = Voice_Recognizer(os.path.join("data_files", "voice_training"))
    voice_rec.full_research()
    """
    for file in glob.glob(os.path.join("demo", "*.wav")):
        print(os.path.basename(file) + " " + voice_rec.predict(file))
    """
    
if __name__ == "__main__":
    main()