import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
# from pydub import AudioSegment, silence
import speech_recognition as sr_module
import os

st.set_page_config(page_title="SpeakSmart", layout="centered")
st.title("🎤 Speech Analysis")

uploaded_file = st.file_uploader("Upload your .wav or .mp3 file", type=["wav", "mp3"])

xgb = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

def extract_wpm(file_path):
    y, sample_rate = librosa.load(file_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sample_rate)

    recognizer = sr_module.Recognizer()
    with sr_module.AudioFile(file_path) as source:
        try:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            word_count = len(text.split())
            wpm = (word_count / duration) * 60
        except:
            wpm = 0
    return wpm

def categorize_wpm(wpm):
    if wpm < 130:
        return "Slow"
    elif 130 <= wpm <= 165:
        return "Normal"
    else:
        return "Fast"

# emotion_map = {
#     "happy": "positive",
#     "surprised": "positive",
#     "neutral": "neutral",
#     "calm": "neutral",
#     "sad": "negative",
#     "angry": "negative",
#     "fearful": "negative",
#     "disgust": "negative"
# }

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp.wav")


    st.subheader("🔍 Analyzing Speech Speed...")
    wpm = extract_wpm("temp.wav")
    category = categorize_wpm(wpm)

    st.markdown(f"- **Speech Speed (WPM)**: `{round(wpm)}`")
    st.markdown(f"- **Speech Speed Category**: **{category}**")

    st.subheader("😊 Analyzing Sentiment...")

try:
    features = extract_features("temp.wav")
    features_scaled = scaler.transform(features.reshape(1, -1))

    pred = xgb.predict(features_scaled)
    pred_sentiment = le.inverse_transform(pred)[0]  

    # ⚡ Define feedback messages
    feedback_messages = {
        "positive": "Great job! You sound confident and engaging. 🚀",
        "negative": "Try to stay calm and positive, it will improve your delivery. 🌱"
    }

    st.markdown(f"- **Predicted sentiment**: `{pred_sentiment}`")

    # ⚡ Show corresponding feedback
    if pred_sentiment in feedback_messages:
        st.markdown(f"- **Feedback**: {feedback_messages[pred_sentiment]}")

except Exception as e:
    st.error(f"Error in sentiment analysis: {e}")
