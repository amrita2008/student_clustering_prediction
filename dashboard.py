import streamlit as st
import requests

st.set_page_config(page_title="Student Clustering Dashboard", layout="centered")

st.title(" Student Performance Clustering Dashboard")
st.write("Predict student cluster using a deployed ML model.")
API_URL = "https://student-clustering-prediction.onrender.com/predict"


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
        response = requests.post(API_URL, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        st.success(f" Predicted Cluster: {result['cluster']}")
    except Exception as e:
        st.error(f"API Error: {e}")


        
  
