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
model_ready = False


def train_model():
    global scaler, kmeans, features, model_ready

    df = pd.read_csv(DATA_FILE)

    # Encode categorical columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    X = df.select_dtypes(include=["int64", "float64"])
    features = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    os.makedirs(MODEL_DIR, exist_ok=True)
    pickle.dump(scaler, open(f"{MODEL_DIR}/scaler.pkl", "wb"))
    pickle.dump(kmeans, open(f"{MODEL_DIR}/kmeans.pkl", "wb"))
    pickle.dump(features, open(f"{MODEL_DIR}/features.pkl", "wb"))

    model_ready = True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": model_ready
    }


@app.post("/train")
def train():
    global model_ready

    if model_ready:
        return {"status": "already trained"}

    train_model()
    return {"status": "training completed"}


@app.post("/predict")
def predict(student: dict):
    if not model_ready:
        return {"error": "Model not trained yet"}

    values = [student.get(f, 0) for f in features]
    arr = np.array(values).reshape(1, -1)
    scaled = scaler.transform(arr)
    cluster = int(kmeans.predict(scaled)[0])

    return {"cluster": cluster}

