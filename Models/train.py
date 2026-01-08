import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import umap

# Load data
df = pd.read_csv("xAPI-Edu-Data.csv")

# Select numeric features
X = df.select_dtypes(include="number")

# Normalize / scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality reduction
umap_model = umap.UMAP(n_components=5, random_state=42)
X_emb = umap_model.fit_transform(X_scaled)

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_emb)

print("Silhouette Score:", silhouette_score(X_emb, labels))

# Hierarchical
hier_model = AgglomerativeClustering(n_clusters=4)
hier_labels = hier_model.fit_predict(X_emb)

# Save models
with open("models/scaler.pkl", "wb") as f: pickle.dump(scaler, f)
with open("models/umap.pkl", "wb") as f: pickle.dump(umap_model, f)
with open("models/kmeans.pkl", "wb") as f: pickle.dump(kmeans, f)
with open("models/hierarchical.pkl", "wb") as f: pickle.dump(hier_model, f)

print("Training complete and models saved!")

