import streamlit as st
import requests
import time

st.title("Student Performance Clustering Dashboard")

API_BASE = "https://student-clustering-prediction.onrender.com"

gender = st.selectbox("Gender", [0, 1])
raisedhands = st.slider("Raised Hands", 0, 100, 10)
visited_resources = st.slider("Visited Resources", 0, 100, 20)
announcements = st.slider("Announcements Viewed", 0, 100, 5)
discussion = st.slider("Discussion Participation", 0, 100, 10)

if st.button("Predict Cluster"):
    payload = {
        "gender": gender,
        "raisedhands": raisedhands,
        "VisITedResources": visited_resources,
        "AnnouncementsView": announcements,
        "Discussion": discussion
    }

    try:
        health = requests.get(f"{API_BASE}/health", timeout=60).json()

        if not health["model_ready"]:
            st.warning("Model is training. Please try again in 1 minute.")
            st.stop()

        response = requests.post(
            f"{API_BASE}/predict",
            json=payload,
            timeout=20
        )
        result = response.json()
        st.success(f" Predicted Cluster: {result['cluster']}")

    except Exception as e:
        st.error(f"Error: {e}")





        
  
