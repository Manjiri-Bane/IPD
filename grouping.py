import pandas as pd

# Load your CSV
df = pd.read_csv("cleaned_features.csv")

# Define mapping
emotion_map = {
    "happy": "positive",
    "surprised": "positive",
    "neutral": "neutral",
    "calm": "neutral",
    "sad": "negative",
    "angry": "negative",
    "fearful": "negative",
    "disgust": "negative"
}

# Apply mapping to create new column
df["sentiment"] = df["emotion"].map(emotion_map)

# Save as new CSV
df.to_csv("cleaned_features_grouped.csv", index=False)

print("✅ File saved as cleaned_features_grouped.csv")
