# data.py
import streamlit as st
import pandas as pd

def app():
    st.title("📊 Data Overview")

    try:
        df = pd.read_csv("Cleaned_indian_air_data.csv", parse_dates=["Date"])
    except:
        st.error("❌ Could not load Cleaned_indian_air_data.csv")
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
