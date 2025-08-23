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

# Path to folder containing WAV files
folder_path = "C:/Users/manji/OneDrive/Desktop/New folder/RAVDESS"

# Load CSV with file names and sentiments
df = pd.read_csv("cleaned_features_grouped.csv")  # Columns: 'file_name' and 'sentiment'

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    return features

# Get unique file names to avoid duplicates
unique_files = df["file_name"].unique()

X = []
y = []

for i, f in enumerate(unique_files):
    file_path = os.path.join(folder_path, f)
    print(f"Processing {i+1}/{len(unique_files)}: {f}")
    try:
        feats = extract_features(file_path)
        X.append(feats)
        # Take the first sentiment if duplicates exist
        sentiment = df[df["file_name"] == f]["sentiment"].iloc[0]
        y.append(sentiment)
    except Exception as e:
        print(f"⚠️ Skipped {f} due to error: {e}")

X = np.array(X)
y = np.array(y)

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------- Model 1: Random Forest -----------
rf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n🔹 Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred, target_names=le.classes_))

# ----------- Model 2: XGBoost -----------
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

# Save models and preprocessing objects
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(xgb, "xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\n✅ Models, Scaler, and LabelEncoder saved successfully!")

'''
import librosa
import numpy as np

# Path to the single WAV file you want to test
file_path = "C:/Users/manji/OneDrive/Desktop/New folder/RAVDESS/03-01-01-01-01-01-01.wav"

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    return features

# Extract features for this single file
X_single = extract_features(file_path)
print("Features shape:", X_single.shape)
'''
'''
import pandas as pd
df = pd.read_csv("cleaned_features_grouped.csv")  # Columns: 'file_name' and 'sentiment'

print("Total rows in CSV:", len(df))
print("Unique file names:", df['file_name'].nunique())
'''
'''
# After splitting and scaling
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")
'''