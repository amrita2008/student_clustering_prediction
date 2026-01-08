import os
import pickle
import pandas as pd
import numpy as np

from fastapi import FastAPI
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = FastAPI()

MODEL_DIR = "models"
DATA_FILE = "xAPI-Edu-Data.csv"

scaler = None
kmeans = None
features = None


def train_if_needed():
    global scaler, kmeans, features

    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(f"{MODEL_DIR}/kmeans.pkl"):
        scaler = pickle.load(open(f"{MODEL_DIR}/scaler.pkl", "rb"))
        kmeans = pickle.load(open(f"{MODEL_DIR}/kmeans.pkl", "rb"))
        features = pickle.load(open(f"{MODEL_DIR}/features.pkl", "rb"))
        return

    df = pd.read_csv(DATA_FILE)

    # encode categoricals
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    X = df.select_dtypes(include=["int64", "float64"])
    features = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    pickle.dump(scaler, open(f"{MODEL_DIR}/scaler.pkl", "wb"))
    pickle.dump(kmeans, open(f"{MODEL_DIR}/kmeans.pkl", "wb"))
    pickle.dump(features, open(f"{MODEL_DIR}/features.pkl", "wb"))


@app.on_event("startup")
def startup_event():
    train_if_needed()


@app.post("/predict")
def predict(student: dict):
    values = [student[f] for f in features]
    arr = np.array(values).reshape(1, -1)
    scaled = scaler.transform(arr)
    cluster = int(kmeans.predict(scaled)[0])
    return {"cluster": cluster}
