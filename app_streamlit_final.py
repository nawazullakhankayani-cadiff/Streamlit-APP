# Streamlit app: EDA + Train Multiple Linear Regression + Predict
# Save and run with: streamlit run app_streamlit_final.py

import os
import time
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# ✅ CHANGED: import multiple linear regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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

# ✅ CHANGED: removed long title text
st.set_page_config(page_title="Air Quality Explorer", layout="wide")

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
    features = [c for c in ['PM2.5','PM10','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene'] if c in df.columns]
    return features, 'AQI'

# ✅ CHANGED: train MULTIPLE linear regression models
def train_multiple_models(df, features, target):
    X = df[features]
    y = df[target]

    mask = (~y.isna()) & (X.notna().sum(axis=1) > 0)
    X, y = X.loc[mask], y.loc[mask]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5)
    }

    results = []

    for name, model in models.items():
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)

        rmse = np.sqrt(mean_squared_error(yte, preds))
        r2 = r2_score(yte, preds)

        path = os.path.join(MODEL_FOLDER, f"{name}_{STUDENT_ID}_{int(time.time())}.pkl")
        joblib.dump(pipe, path)

        results.append({
            "Model": name,
            "RMSE": round(rmse, 3),
            "R2": round(r2, 3),
            "Path": path
        })

    return pd.DataFrame(results)

# ---------------- Sidebar ----------------
page = st.sidebar.radio(
    "Choose page",
    ["🏠 Home", "📊 Data", "🔍 EDA", "🧠 Train ML Model", "🔮 Predict", "⬇️ Download", "📚 Reference"]
)

st.sidebar.write(f"Student: **{STUDENT_NAME}**")
st.sidebar.write(f"ID: **{STUDENT_ID}**")

uploaded_csv = st.sidebar.file_uploader("Upload cleaned CSV", type=["csv"])
use_local = st.sidebar.checkbox("Use local CSV", value=True)

# ---------------- Load Data ----------------
if uploaded_csv:
    df = load_csv_file(uploaded_csv)
elif use_local and os.path.exists(LOCAL_CSV):
    df = load_csv_file(LOCAL_CSV)
else:
    df = pd.DataFrame()

# ---------------- Pages ----------------
if page == "🏠 Home":
    st.header("Indian Air Quality Data Explorer")
    if not df.empty:
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])

elif page == "📊 Data":
    st.header("Dataset Preview")
    st.dataframe(df.head(20))

elif page == "🔍 EDA":
    st.header("Exploratory Data Analysis")
    col = st.selectbox("Select column", df.select_dtypes(include=np.number).columns)
    fig = px.histogram(df, x=col, nbins=50)
    st.plotly_chart(fig, use_container_width=True)

elif page == "🧠 Train ML Model":
    st.header("Train Multiple Linear Regression Models")
    if st.button("Train Models"):
        features, target = prepare_features_and_target(df)
        results = train_multiple_models(df, features, target)
        st.dataframe(results)
        best = results.sort_values("RMSE").iloc[0]
        st.success(f"Best Model: {best['Model']} | RMSE={best['RMSE']} | R²={best['R2']}")

elif page == "🔮 Predict":
    st.header("Predict AQI")
    model_files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pkl")]
    model_name = st.selectbox("Select model", model_files)
    model = joblib.load(os.path.join(MODEL_FOLDER, model_name))

    features, _ = prepare_features_and_target(df)
    inputs = {f: st.number_input(f, value=0.0) for f in features}

    if st.button("Predict AQI"):
        pred = model.predict(pd.DataFrame([inputs]))[0]
        st.metric("Predicted AQI", f"{pred:.2f}")

elif page == "⬇️ Download":
    st.header("Download Models")
    for f in os.listdir(MODEL_FOLDER):
        with open(os.path.join(MODEL_FOLDER, f), "rb") as fh:
            st.download_button(f, fh, file_name=f)

# ---------------- Reference ----------------
elif page == "📚 Reference":
    st.header("Model Reference & Inference")
    st.markdown("""
This application performs exploratory data analysis and implements **multiple linear
regression algorithms** (Linear Regression, Ridge, Lasso, and ElasticNet) to predict
the **Air Quality Index (AQI)**.

All models are trained using a standardized pipeline that includes missing-value
imputation and feature scaling. Model performance is evaluated using **RMSE** and
**R² metrics**.

The observed RMSE values indicate **moderate prediction accuracy**, which is expected
since AQI depends on nonlinear, seasonal, and city-specific factors that simple linear
models cannot fully capture. These models therefore act as **baseline predictors**.
""")
