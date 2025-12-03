# home.py
import streamlit as st
import pandas as pd

def app():
    st.title("🌎 AirLens – Indian Air Quality Analysis")

    st.markdown("""
    ### Welcome to AirLens
    This application analyzes Indian air quality, visualizes pollutant levels,
    and trains machine learning models to predict AQI.

    **Pages included:**
    - 📊 Data Overview  
    - 📈 EDA & Visualizations  
    - 🤖 Train Machine Learning Model  
    - 🔮 Predict AQI  
    """)

    uploaded_img = st.file_uploader("Upload a header image (optional)", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        st.image(uploaded_img, use_column_width=True)

