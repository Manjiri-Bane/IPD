import librosa
import numpy as np
import joblib
import os

# Load XGBoost model and preprocessing objects
xgb = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

# Hardcoded WAV files
wav_files = [
    "C:/Users/manji/OneDrive/Desktop/New folder/RAVDESS/03-01-01-01-01-01-01.wav",
    "C:/Users/manji/OneDrive/Desktop/New folder/RAVDESS/03-01-01-01-01-02-01.wav"
]

for file_path in wav_files:
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        continue

    # Extract and scale features
    features = extract_features(file_path)
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict class and probabilities
    pred = xgb.predict(features_scaled)
    proba = xgb.predict_proba(features_scaled)[0]

    label = le.inverse_transform(pred)[0]
    print(f"\nFile: {os.path.basename(file_path)}")
    print(f"🔹 Predicted Class: {label}")
    print("🔹 Class Probabilities:")
    for cls, p in zip(le.classes_, proba):
        print(f"   {cls}: {p:.3f}")
