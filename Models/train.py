import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir())
DATA_FILE = "xAPI-Edu-Data.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found in container")

df = pd.read_csv(DATA_FILE)
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("category").cat.codes
X = df.select_dtypes(include=["int64", "float64"])

if X.shape[1] == 0:
    raise ValueError("No numeric features found after preprocessing")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", score)
os.makedirs("models", exist_ok=True)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("models/features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)
