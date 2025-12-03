# prediction.py
import streamlit as st
import pandas as pd
import joblib
import os

MODEL_FOLDER = "models_st20329043"

def app():
    st.title("🔮 Predict AQI")

    st.markdown("Load a trained Linear Regression model and enter pollutant values.")

    # Load latest trained model automatically
    model_files = sorted(
        [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pkl")],
        reverse=True
    )

    if len(model_files) == 0:
        st.error("❌ No trained model found. Train a model first.")
        return

    latest_model = os.path.join(MODEL_FOLDER, model_files[0])

    st.info(f"Loaded model: **{latest_model}**")

    model = joblib.load(latest_model)

    # Input form
    st.subheader("Enter pollutant values")

    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]
    inputs = {}

    for p in pollutants:
        inputs[p] = st.number_input(p, min_value=0.0, value=10.0)

    if st.button("Predict AQI"):
        df = pd.DataFrame([inputs])

        # Pipeline handles imputation + scaling automatically
        pred = model.predict(df)[0]

        st.success(f"### 🌬 Predicted AQI: **{pred:.2f}**")
