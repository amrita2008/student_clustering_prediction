from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()
with open("models/scaler.pkl","rb") as f: scaler = pickle.load(f)
with open("models/umap.pkl","rb") as f: umap_model = pickle.load(f)
with open("models/kmeans.pkl","rb") as f: kmeans = pickle.load(f)

class StudentData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(student: StudentData):
    arr = np.array(student.features).reshape(1, -1)
    scaled = scaler.transform(arr)
    emb = umap_model.transform(scaled)
    cluster = int(kmeans.predict(emb)[0])
    return {"cluster": cluster}
