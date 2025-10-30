import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("initial.csv")
emotion_counts = df['emotion'].value_counts().sort_index()  

colors = ['#1f77b4',  
          '#ff7f0e', 
          '#2ca02c',
          '#d62728',  
          '#9467bd', 
          '#8c564b', 
          '#e377c2', 
          '#17becf']  

plt.figure(figsize=(10,6))
plt.bar(emotion_counts.index, emotion_counts.values, color=colors)
plt.title('Number of Samples per Emotion', fontsize=16)
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
