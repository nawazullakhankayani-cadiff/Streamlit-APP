# app.py  ---------------------------------------------------------------
# Single-file Streamlit Air Quality App (Home + Data + EDA + Train + Predict)
# Works with: Cleaned_indian_air_data.csv
# Author: Your Name

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
CSV_PATH = "Cleaned_indian_air_data.csv"
MODEL_DIR = "models_st20329043"
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="AirLens – Indian Air Quality", layout="wide")


# ------------------------------------------------------------
# LOAD DATA FUNCTION
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"], dayfirst=True)
    df.columns = [c.strip() for c in df.columns]
    # Convert numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df["YEAR"] = df["Date"].dt.year
    df["MONTH"] = df["Date"].dt.month
    return df


# ------------------------------------------------------------
# PAGE NAVIGATION
# ------------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Data Overview",
        "📈 EDA",
        "🤖 Train Model",
        "🔮 Predict AQI"
    ]
)


# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------
if page == "🏠 Home":
    st.title("🌏 AirLens – Indian Air Quality Dashboard")
    st.markdown("""
    This application lets you explore Indian Air Quality data,
    visualize pollutants, train ML models, and predict AQI.
    
    **Sections included:**
    - 📊 Data Overview  
    - 📈 EDA  
    - 🤖 Train Regression Model  
    - 🔮 Predict AQI  
    """)


# ------------------------------------------------------------
# DATA OVERVIEW
# ------------------------------------------------------------
elif page == "📊 Data Overview":
    st.title("📊 Data Overview")

    try:
        df = load_data()
    except:
        st.error("❌ Could not load Cleaned_indian_air_data.csv")
        st.stop()

    st.subheader("Preview (Top 10 Rows)")
    st.dataframe(df.head(10))

    st.subheader("Missing Values Summary")
    missing = df.isnull().sum().sort_values(ascending=False)
    miss_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": (missing / len(df) * 100).round(2)
    })
    st.dataframe(miss_df)


# ------------------------------------------------------------
# EDA PAGE
# ------------------------------------------------------------
elif page == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")

    try:
        df = load_data()
    except:
        st.error("❌ Dataset missing.")
        st.stop()

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    pollutant = st.selectbox("Choose pollutant", numeric_cols)

    st.subheader(f"{pollutant} Trend Over Time")
    fig = px.line(df.sort_values("Date"), x="Date", y=pollutant, color="City")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig2 = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig2, use_container_width=True)


# ------------------------------------------------------------
# TRAIN MODEL PAGE
# ------------------------------------------------------------
elif page == "🤖 Train Model":
    st.title("🤖 Train AQI Prediction Model")

    try:
        df = load_data()
    except:
        st.error("Dataset missing.")
        st.stop()

    features = [
        col for col in df.columns
        if col not in ["AQI", "City", "Date", "AQI_Bucket"]
    ]
    target = "AQI"

    st.write("Features:", features)

    if st.button("Train Linear Regression Model"):
        X = df[features]
        y = df[target]

        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        model_path = os.path.join(MODEL_DIR, f"aqi_model_{int(time.time())}.pkl")
        joblib.dump(pipe, model_path)

        st.success("Model trained successfully!")
        st.write("📉 RMSE:", rmse)
        st.write("📈 R² Score:", r2)
        st.write("📁 Model saved at:", model_path)


# ------------------------------------------------------------
# PREDICT PAGE
# ------------------------------------------------------------
elif page == "🔮 Predict AQI":
    st.title("🔮 Predict AQI")

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

    if not model_files:
        st.error("❌ No trained model found. Train one first.")
        st.stop()

    model_name = st.selectbox("Choose a model", model_files)
    model = joblib.load(os.path.join(MODEL_DIR, model_name))

    st.subheader("Enter pollutant values")

    # Automatic numeric feature detection
    df = load_data()
    features = [
        col for col in df.columns
        if col not in ["AQI", "City", "Date", "AQI_Bucket"]
    ]

    inputs = {}

    cols = st.columns(3)
    for i, feat in enumerate(features):
        with cols[i % 3]:
            inputs[feat] = st.number_input(feat, min_value=0.0, value=50.0)

    if st.button("Predict AQI"):
        X = pd.DataFrame([inputs])
        pred = model.predict(X)[0]

        st.success(f"### 🌬 Predicted AQI: **{pred:.2f}**")
