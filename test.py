import librosa
import numpy as np
import joblib
import os

# Load XGBoost model and preprocessing objects
xgb = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

# Hardcoded WAV file path
file_path = "C:/Users/manji/Downloads/Neutral.wav"

if not os.path.exists(file_path):
    print("⚠️ File not found!")
    exit()

# Extract features, scale, and reshape
features = extract_features(file_path)
features_scaled = scaler.transform(features.reshape(1, -1))

# Predict
pred = xgb.predict(features_scaled)
label = le.inverse_transform(pred)[0]

print(f"\n🔹 XGBoost Prediction: {label}")
