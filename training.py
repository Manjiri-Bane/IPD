import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

folder_path = r"D:\Minori Wakade\3rd sem\IPD\ml-public-speaking-evaluator\IPD\RAVDESS"

df = pd.read_csv("grouped.csv")  # Columns: 'file_name' and 'sentiment'

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    return features

unique_files = df["file_name"].unique()

X = []
y = []

for i, f in enumerate(unique_files):
    file_path = os.path.join(folder_path, f)
    print(f"Processing {i+1}/{len(unique_files)}: {f}")
    try:
        feats = extract_features(file_path)
        X.append(feats)
        sentiment = df[df["file_name"] == f]["sentiment"].iloc[0]
        y.append(sentiment)
    except Exception as e:
        print(f"⚠️ Skipped {f} due to error: {e}")

X = np.array(X)
y = np.array(y)

feature_names = [f"feat_{i}" for i in range(X.shape[1])]
features_df = pd.DataFrame(X, columns=feature_names)
features_df["file_name"] = unique_files[:len(X)]  # to keep track
features_df["sentiment"] = y

features_df.to_csv("extracted_features.csv", index=False)
print("Saved extracted features to extracted_features.csv")

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n🔹 Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred, target_names=le.classes_))

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("\n🔹 XGBoost Results:")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred, target_names=le.classes_))

joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(xgb, "xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\n✅ Models, Scaler, and LabelEncoder saved successfully!")