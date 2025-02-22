from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import json

# Set Streamlit page config for better UI
st.set_page_config(page_title="Car Damage Estimator", layout="wide")

# Custom CSS for styling (dark theme, stylish buttons)
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
            font-family: Arial, sans-serif;
        }
        .stButton button {
            background-color: #cba135;
            color: black;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stImage img {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Banner Image
st.markdown("<h1 style='text-align: center;'>🚗 Repair Cost Car Damage Estimation</h1>", unsafe_allow_html=True)

# Load the model
load_model = YOLO(".../best.pt")

# Upload File Section
uploaded_file = st.file_uploader("Upload an image of your damaged car", type=["jpg", "jpeg", "png"])

# Load damage price data
with open("car_damage_price.json", "r") as file:
    data = json.load(file)

if uploaded_file is not None:
    
    # Split into two columns for better layout
    col1, col2 = st.columns(2)

    # Open image using PIL
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
    # Convert PIL image to NumPy array
    image_array = np.array(image)
    
    results = load_model(image_array)

    if results and results[0].boxes is not None and hasattr(results[0].boxes, "cls"):
        detected_classes = [results[0].names[int(cls)] for cls in results[0].boxes.cls.tolist()]
        
        with col2:
            st.image(results[0].plot(), caption="Detection Result", use_column_width=True)

        st.markdown("## 🔍 Detected Damages")
        
        found = False
        for cls in detected_classes:
            if cls in data:
                st.markdown(f"✅ **Estimated Repair Cost**: **${data[cls]}**")
                found = True
        
        if not found:
            st.warning("⚠️ Please upload a closer image for better detection.")

# Footer Section
st.markdown("""
    <br><br>
    <hr style="border: 1px solid #cba135;">
    <p style="text-align: center; color: #cba135;">
        🚗 Powered by AI | Car Damage Estimation System
    </p>
""", unsafe_allow_html=True)

