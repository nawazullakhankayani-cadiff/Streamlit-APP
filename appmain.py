# app.py  (Single File Streamlit App - All Pages Combined)

import streamlit as st
import pandas as pd
import os
import joblib
import time
import plotly.express as px

# ----------------------------
# Global Settings
# ----------------------------
CSV_PATH = "final_cleaned_air_quality.csv"
MODEL_FOLDER = "models_st20329043"
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ----------------------------
# HOME PAGE
# ----------------------------
def home_page():
    st.title("🌎 AirLens – Indian Air Quality Analysis")

    st.markdown("""
    ### Welcome to AirLens  
    Explore Indian air pollution levels, view pollutant trends,  
    train ML models, and predict AQI values using real-time inputs.

    **Pages included:**
    - 📊 Data Overview  
    - 📈 EDA  
    - 🤖 Train ML Model  
    - 🔮 Predict AQI  
    """)

    img = st.file_uploader("Upload a header image (optional)", type=["png", "jpg", "jpeg"])
    if img:
        st.image(img, use_column_width=True)

# ----------------------------
# DATA PAGE
# ----------------------------
def data_page():
    st.title("📊 Data Overview")

    try:
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    except:
        st.error("❌ ERROR: Could not load final_cleaned_air_quality.csv")
        return

    st.subheader("Preview (Top 10 Rows)")
    st.dataframe(df.head(10))

    st.subheader("Missing Values Summary")
    missing = df.isnull().sum().sort_values(ascending=False)
    miss_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": (missing / len(df) * 100).round(2)
    })
    st.dataframe(miss_df)

# ----------------------------
# EDA PAGE
# ----------------------------
def eda_page():
    st.title("📈 Exploratory Data Analysis (EDA)")

    try:
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    except:
        st.error("❌ Could not load final_cleaned_air_quality.csv")
        return

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    st.subheader("Pollutant Trend Over Time")
    col = st.selectbox("Select pollutant", numeric_cols)

    fig = px.line(df.sort_values("Date"), x="Date", y=col,
                  title=f"{col} Trend Over Time")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig2 = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# TRAIN MODEL PAGE
# ----------------------------
def model_train_page():
    st.title("🤖 Train Machine Learning Model")

    if not os.path.exists(CSV_PATH):
        st.error("❌ Dataset not found!")
        return

    df = pd.read_csv(CSV_PATH)

    # Remove unwanted columns
    features = [col for col in df.columns if col not in ["AQI", "City", "Date"]]
    target = "AQI"

    st.write("📌 Features used for training:")
    st.write(features)

    if st.button("Train Linear Regression Model"):
        from sklearn.model_selection import train_test_split
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score

        X = df[features]
        y = df[target]

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        model_path = os.path.join(MODEL_FOLDER, f"aqi_model_{int(time.time())}.pkl")
        joblib.dump(pipeline, model_path)

        st.success("🎉 Model trained successfully!")
        st.write(f"📉 RMSE: **{rmse:.3f}**")
        st.write(f"📈 R² Score: **{r2:.3f}**")
        st.write(f"📁 Saved model: `{model_path}`")

# ----------------------------
# PREDICTION PAGE
# ----------------------------
def prediction_page():
    st.title("🔮 Predict AQI")

    # Load latest model
    model_files = sorted(
        [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pkl")],
        reverse=True
    )

    if len(model_files) == 0:
        st.error("❌ No model found! Please train a model first.")
        return

    latest_model = os.path.join(MODEL_FOLDER, model_files[0])
    model = joblib.load(latest_model)

    st.info(f"Loaded model: **{latest_model}**")

    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]
    values = {}

    st.subheader("Enter Pollutant Values")
    for p in pollutants:
        values[p] = st.number_input(p, min_value=0.0, value=10.0)

    if st.button("Predict AQI"):
        df = pd.DataFrame([values])
        pred = model.predict(df)[0]
        st.success(f"### 🌬 Predicted AQI: **{pred:.2f}**")

# ----------------------------
# SIDEBAR NAVIGATION
# ----------------------------
pages = {
    "🏠 Home": home_page,
    "📊 Data Overview": data_page,
    "📈 EDA": eda_page,
    "🤖 Train ML Model": model_train_page,
    "🔮 Predict AQI": prediction_page
}

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", list(pages.keys()))

# Run selected page
pages[choice]()
