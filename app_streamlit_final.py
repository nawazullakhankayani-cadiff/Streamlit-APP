# Streamlit app: EDA + Train Multiple Linear Regression + Predict

import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- Config ----------------
STUDENT_NAME = "Nawazullakhan kayani"
STUDENT_ID = "st20329043"
MODEL_FOLDER = f"models_{STUDENT_ID}"
LOCAL_CSV = "Cleaned_indian_air_data (15).csv"

os.makedirs(MODEL_FOLDER, exist_ok=True)

st.set_page_config(page_title="Air Quality Explorer", layout="wide")
st.title("Air Quality Explorer")

# ---------------- Helpers ----------------
@st.cache_data
def load_csv_file(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    for col in ['PM2.5','PM10','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene','AQI']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        df['YEAR'] = df['Date'].dt.year
        df['MONTH'] = df['Date'].dt.month

    return df


def prepare_features_and_target(df):
    features = [c for c in [
        'PM2.5','PM10','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene'
    ] if c in df.columns]
    target = 'AQI'
    return features, target


def train_multiple_linear(df, features, target):
    X = df[features]
    y = df[target]

    mask = (~y.isna()) & (X.notna().sum(axis=1) > 0)
    X = X.loc[mask]
    y = y.loc[mask]

    if len(X) < 30:
        raise ValueError("Not enough data for training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    ts = int(time.time())
    model_path = os.path.join(
        MODEL_FOLDER, f"multiple_linear_aqi_{STUDENT_ID}_{ts}.pkl"
    )
    joblib.dump(pipeline, model_path)

    coef = pipeline.named_steps["model"].coef_
    coef_map = dict(zip(features, coef))

    return pipeline, model_path, rmse, r2, coef_map


# ---------------- Sidebar ----------------
page = st.sidebar.radio(
    "Choose page",
    ["🏠 Home", "📊 Data", "🔍 EDA", "🧠 Train ML Model", "🔮 Predict", "⬇️ Download"]
)

st.sidebar.header("Project Info")
st.sidebar.write(f"Student: **{STUDENT_NAME}**")
st.sidebar.write(f"ID: **{STUDENT_ID}**")

uploaded_csv = st.sidebar.file_uploader("Upload cleaned CSV", type=["csv"])
use_local = st.sidebar.checkbox("Use local CSV if available", value=True)

# ---------------- Load Data ----------------
if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
else:
    df = load_csv_file(LOCAL_CSV) if use_local else pd.DataFrame()

# ---------------- Home ----------------
if page == "🏠 Home":
    st.header("Welcome to the Indian Air Quality Data Explorer")
    if df.empty:
        st.warning("No dataset loaded.")
    else:
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])

# ---------------- Data ----------------
elif page == "📊 Data":
    st.header("Dataset Preview")
    if df.empty:
        st.warning("No data loaded.")
    else:
        st.dataframe(df.head(10))
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())

# ---------------- EDA (UNCHANGED) ----------------
elif page == "🔍 EDA":
    st.header("Exploratory Data Analysis")
    if df.empty:
        st.warning("No data loaded.")
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns
        selected = st.multiselect(
            "Select numeric columns",
            numeric_cols,
            default=list(numeric_cols[:3])
        )

        if selected:
            fig = px.box(df, y=selected, title="EDA Boxplot")
            st.plotly_chart(fig, use_container_width=True)

# ---------------- Train Model ----------------
elif page == "🧠 Train ML Model":
    st.header("Train Multiple Linear Regression Model to Predict AQI")

    if df.empty:
        st.warning("Load data first.")
    else:
        features, target = prepare_features_and_target(df)
        st.write("Features:", features)

        if st.button("Train Multiple Linear Regression Model"):
            try:
                model, path, rmse, r2, coef = train_multiple_linear(df, features, target)
                st.success("Model trained successfully!")
                st.write("Model saved at:", path)
                st.metric("RMSE", round(rmse, 2))
                st.metric("R² Score", round(r2, 2))
                st.subheader("Model Coefficients (Multiple Linear Regression)")
                st.dataframe(pd.DataFrame.from_dict(coef, orient="index", columns=["Coefficient"]))
                st.session_state["model"] = model
            except Exception as e:
                st.error(e)

# ---------------- Predict ----------------
elif page == "🔮 Predict":
    st.header("AQI Prediction using Multiple Linear Regression")

    if "model" not in st.session_state:
        st.info("Train a model first.")
    else:
        model = st.session_state["model"]
        features, _ = prepare_features_and_target(df)

        inputs = {}
        for f in features:
            inputs[f] = st.number_input(f, value=float(df[f].median()))

        if st.button("Predict AQI"):
            X_new = pd.DataFrame([inputs])
            pred = model.predict(X_new)[0]
            st.metric("Predicted AQI", round(pred, 2))

# ---------------- Download ----------------
elif page == "⬇️ Download":
    st.header("Download Models")
    files = os.listdir(MODEL_FOLDER)
    for f in files:
        with open(os.path.join(MODEL_FOLDER, f), "rb") as file:
            st.download_button(f, file, f)

