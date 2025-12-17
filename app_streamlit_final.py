import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# ================= PAGE CONFIG (WHITE BACKGROUND) =================
st.set_page_config(page_title="Air Quality Explorer", layout="wide")

st.markdown("""
<style>
body {
    background-color: white;
    color: black;
}
[data-testid="stAppViewContainer"] {
    background-color: white;
}
[data-testid="stSidebar"] {
    background-color: #f5f5f5;
}
h1, h2, h3, h4, h5 {
    color: black;
}
</style>
""", unsafe_allow_html=True)

# ================= CONFIG =================
STUDENT_NAME = "Nawazullakhan kayani"
STUDENT_ID = "st20329043"
MODEL_FOLDER = f"models_{STUDENT_ID}"
LOCAL_CSV = "Cleaned_indian_air_data (15).csv"

os.makedirs(MODEL_FOLDER, exist_ok=True)

# ================= HELPERS =================
@st.cache_data
def load_data(source):
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]

    num_cols = [
        'PM2.5','PM10','NO2','NOx','NH3','CO',
        'SO2','O3','Benzene','Toluene','Xylene','AQI'
    ]

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['YEAR'] = df['Date'].dt.year
        df['MONTH'] = df['Date'].dt.month

    return df

def get_features_target(df):
    features = [
        c for c in [
            'PM2.5','PM10','NO2','NOx','NH3','CO',
            'SO2','O3','Benzene','Toluene','Xylene'
        ] if c in df.columns
    ]
    return features, 'AQI'

def train_model(df, model, name, features, target):
    X, y = df[features], df[target]
    mask = (~y.isna()) & (X.notna().sum(axis=1) > 0)
    X, y = X.loc[mask], y.loc[mask]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)

    rmse = np.sqrt(mean_squared_error(yte, preds))
    r2 = r2_score(yte, preds)

    path = os.path.join(
        MODEL_FOLDER, f"{name}_{STUDENT_ID}_{int(time.time())}.pkl"
    )
    joblib.dump(pipe, path)

    return {
        "Model": name,
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3),
        "Train Rows": len(Xtr),
        "Test Rows": len(Xte),
        "Path": path
    }

# ================= SIDEBAR =================
page = st.sidebar.radio(
    "Choose page",
    ["🏠 Home", "📊 Data", "🔍 EDA", "🧠 Train Models", "🔮 Predict", "📚 Reference", "⬇️ Download"]
)

st.sidebar.markdown(f"**Student:** {STUDENT_NAME}")
st.sidebar.markdown(f"**ID:** {STUDENT_ID}")

uploaded_csv = st.sidebar.file_uploader("Upload cleaned CSV", type=["csv"])
use_local = st.sidebar.checkbox("Use local CSV", value=True)

# ================= LOAD DATA =================
df = pd.DataFrame()
if uploaded_csv:
    df = load_data(uploaded_csv)
elif use_local and os.path.exists(LOCAL_CSV):
    df = load_data(LOCAL_CSV)

# ================= PAGES =================

# ---------- HOME ----------
if page == "🏠 Home":
    st.header("Indian Air Quality Data Explorer")

    images = [f for f in os.listdir(".") if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    for img in images:
        st.image(img, use_column_width=True)

    if images:
        st.markdown("""
<small><i>
Note: The images displayed above are sourced from Google AI and are used strictly for
demonstration and review purposes only. They are included to enhance the visual
presentation of the application and do not influence the data analysis or model results.
</i></small>
""", unsafe_allow_html=True)

    if not df.empty:
        col1, col2 = st.columns(2)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

# ---------- DATA ----------
elif page == "📊 Data":
    st.header("Dataset Overview")
    if df.empty:
        st.warning("No dataset loaded.")
    else:
        st.dataframe(df.head(20))

# ---------- EDA ----------
elif page == "🔍 EDA":
    st.header("Exploratory Data Analysis")
    if df.empty:
        st.warning("Load dataset first.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns
        col = st.selectbox("Select numeric column", num_cols)
        fig = px.histogram(
            df, x=col, nbins=50,
            title=f"Distribution of {col}",
            template="simple_white"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------- TRAIN ----------
elif page == "🧠 Train Models":
    st.header("Train Multiple Linear Regression Models")

    if df.empty:
        st.warning("Load dataset first.")
    else:
        features, target = get_features_target(df)

        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.01),
            "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5)
        }

        if st.button("Train All Models"):
            results = []
            for name, model in models.items():
                results.append(train_model(df, model, name, features, target))

            res_df = pd.DataFrame(results)
            st.subheader("Model Comparison")
            st.dataframe(res_df)

            best = res_df.sort_values("RMSE").iloc[0]
            st.success(
                f"Best Model: {best['Model']} | RMSE={best['RMSE']} | R²={best['R2']}"
            )

# ---------- PREDICT ----------
elif page == "🔮 Predict":
    st.header("Predict AQI")

    model_files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pkl")]
    if not model_files:
        st.warning("No trained models available.")
    else:
        model_name = st.selectbox("Select model", model_files)
        model = joblib.load(os.path.join(MODEL_FOLDER, model_name))

        features, _ = get_features_target(df)
        cols = st.columns(3)
        inputs = {}

        for i, f in enumerate(features):
            with cols[i % 3]:
                inputs[f] = st.number_input(f, min_value=0.0, value=0.0)

        if st.button("Predict AQI"):
            pred = model.predict(pd.DataFrame([inputs]))[0]
            st.metric("Predicted AQI", f"{pred:.2f}")

# ---------- REFERENCE ----------
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

# ---------- DOWNLOAD ----------
elif page == "⬇️ Download":
    st.header("Download Trained Models")

    files = os.listdir(MODEL_FOLDER)
    if not files:
        st.info("No models available.")
    else:
        for f in files:
            with open(os.path.join(MODEL_FOLDER, f), "rb") as fh:
                st.download_button(f, fh, file_name=f)
