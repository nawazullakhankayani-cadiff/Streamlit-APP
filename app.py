# app.py
import streamlit as st
from multiapp import MultiApp

# Import your pages
import home
import data
import eda
import model_train
import prediction

app = MultiApp()

# Add Pages
app.add_app("Home", home.app)
app.add_app("Data Overview", data.app)
app.add_app("EDA", eda.app)
app.add_app("Train ML Model", model_train.app)
app.add_app("Predict AQI", prediction.app)

# Run the app
app.run()
