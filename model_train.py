# model_train.py

import streamlit as st
import pandas as pd
import os
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

CSV_PATH = "final_cleaned_air_quality.csv"
MODEL_FOLDER = "models_st20329043"
os.makedirs(MODEL_FOLDER, exist_ok=True)

def app():
    st.title("🤖 Train Model (Linear Regression)")

    if not os.path.exists(CSV_PATH):
        st.error("Dataset not found.")
        return

    df = pd.read_csv(CSV_PATH)

    features = [col for col in df.columns if col not in ["AQI","City","Date"]]
    target = "AQI"

    st.write("📌 Features used:", features)

    if st.button("Train Model"):
        X = df[features]
        y = df[target]

        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LinearRegression())
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        model_path = os.path.join(MODEL_FOLDER, f"linear_{int(time.time())}.pkl")
        joblib.dump(pipe, model_path)

        st.success("Model trained and saved!")
        st.write("📉 RMSE:", rmse)
        st.write("📈 R² Score:", r2)
        st.write("📁 Saved at:", model_path)
