import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
np.__config__.show()


import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO("yolov8s.pt")

# Initialize session state
if "room_data" not in st.session_state:
    st.session_state.room_data = []

st.set_page_config(page_title="Library Seat Finder", layout="wide")
st.title("📚 AI-Powered Occupancy Finder")

# Theme toggle
theme = st.sidebar.radio("🎨 Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        .main {background-color: #121212; color: #ffffff;}
        </style>
        """, unsafe_allow_html=True)

# View selection
view = st.sidebar.selectbox("Select View", ["Admin", "Student", "Help"])

# Detection functions
def detect_people(image_path):
    results = model(image_path)
    person_detections = [box for box in results[0].boxes.cls if int(box) == 0]
    return len(person_detections), results[0]

def detect_people_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_count, frame_count = 0, 0
    while frame_count < 10:
        ret, frame = cap.read()
        if not ret: break
        frame_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(frame_path, frame)
        count, _ = detect_people(frame_path)
        total_count += count
        frame_count += 1
    cap.release()
    return total_count // frame_count if frame_count else 0

# Admin View
if view == "Admin":
    st.sidebar.header("➕ Add New Room")
    facility_name = st.sidebar.text_input("Facility Name")
    room_name = st.sidebar.text_input("Room Name")
    capacity = st.sidebar.number_input("Room Capacity", min_value=1, value=30)
    image_file = st.sidebar.file_uploader("Upload Room Image", type=["jpg", "jpeg", "png"])
    video_file = st.sidebar.file_uploader("Or Upload Room Video", type=["mp4", "avi", "mov"])

    if st.sidebar.button("Add Room"):
        if (image_file or video_file) and room_name and facility_name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg" if image_file else ".mp4") as temp:
                temp.write((image_file or video_file).read())
                temp_path = temp.name

            occupancy = detect_people(temp_path) if image_file else detect_people_in_video(temp_path)
            occupancy = min(occupancy[0] if image_file else occupancy, capacity)

            st.session_state.room_data.append({
                "Facility": facility_name,
                "Room": room_name,
                "Capacity": capacity,
                "Occupancy": occupancy,
                "Available": capacity - occupancy,
                "ImagePath": temp_path
            })
            st.sidebar.success(f"Room '{room_name}' added successfully!")
        else:
            st.sidebar.error("Provide all required details.")

# Student View
if view == "Student":
    st.sidebar.header("🔍 Room Search")
    search_query = st.sidebar.text_input("Search by Room or Facility")

# Main Dashboard
if view in ["Admin", "Student"] and st.session_state.room_data:
    df = pd.DataFrame(st.session_state.room_data)

    if view == "Student" and search_query:
        df = df[df["Room"].str.contains(search_query, case=False) |
                df["Facility"].str.contains(search_query, case=False)]

    st.dataframe(df["Facility Room Capacity Occupancy Available".split()])

    # Occupancy Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["Room"], df["Occupancy"], color='tomato', label='Occupied')
    ax.barh(df["Room"], df["Available"], left=df["Occupancy"], color='lightgreen', label='Available')
    ax.set_xlabel("Seats")
    ax.set_title("Room Occupancy")
    ax.legend()
    st.pyplot(fig)

# Help Section
if view == "Help":
    st.header("📖 FAQ & Help")
    st.markdown("""
    **Q: How to add rooms?**  
    **A:** Go to Admin view, enter details, upload media, and click "Add Room".

    **Q: How to search rooms?**  
    **A:** Use the sidebar search in Student view to filter by Facility or Room.

    **Q: How often is occupancy updated?**  
    **A:** Admin updates manually when adding new media.

    **Tutorial Video:**
    [Watch here](https://example.com/tutorial)
    """)

# Gamification
if "user_scores" not in st.session_state:
    st.session_state.user_scores = {}

st.sidebar.header("🏅 Leaderboard")
user = st.sidebar.text_input("Enter Username")
if st.sidebar.button("Report Occupancy") and user:
    st.session_state.user_scores[user] = st.session_state.user_scores.get(user, 0) + 10

leaderboard = pd.DataFrame(list(st.session_state.user_scores.items()), columns=["User", "Points"])
leaderboard = leaderboard.sort_values(by="Points", ascending=False).head(5)
st.sidebar.dataframe(leaderboard, use_container_width=True)

st.sidebar.write("🎉 Top contributors earn badges!")
