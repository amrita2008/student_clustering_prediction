import streamlit as st
import requests

st.set_page_config(page_title="Student Clustering Dashboard", layout="centered")

st.title("🎓 Student Performance Clustering Dashboard")
st.write("This dashboard predicts the academic cluster of a student using an ML model.")

# ---- INPUTS ----
gender = st.selectbox("Gender", [0, 1])
raisedhands = st.slider("Raised Hands", 0, 100, 10)
visited_resources = st.slider("Visited Resources", 0, 100, 20)
announcements = st.slider("Announcements Viewed", 0, 100, 5)
discussion = st.slider("Discussion Participation", 0, 100, 10)

# ---- API CALL ----
if st.button("Predict Cluster"):
    payload = {
        "gender": gender,
        "raisedhands": raisedhands,
        "visITedResources": visited_resources,
        "AnnouncementsView": announcements,
        "Discussion": discussion
    }

    try:
        response = requests.post(
            "https://YOUR-API.onrender.com/predict",
            json=payload,
            timeout=10
        )
        result = response.json()
        st.success(f"🎯 Predicted Cluster: {result['cluster']}")

    except Exception as e:
        st.error("API not reachable. Try again later.")
