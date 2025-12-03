# eda.py
import streamlit as st
import pandas as pd
import plotly.express as px

def app():
    st.title("📈 Exploratory Data Analysis (EDA)")

    try:
        df = pd.read_csv("Cleaned_indian_air_data.csv", parse_dates=["Date"])
    except:
        st.error("❌ Could not load Cleaned_indian_air_data.csv")
        return

    st.subheader("Select a Column for Visualization")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    choice = st.selectbox("Choose a pollutant", numeric_cols)

    # Line chart
    st.subheader(f"Trend of {choice} Over Time")
    fig = px.line(df.sort_values("Date"), x="Date", y=choice)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation
    st.subheader("Heatmap — Correlation Between Pollutants")
    corr = df[numeric_cols].corr()
    fig2 = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig2, use_container_width=True)
