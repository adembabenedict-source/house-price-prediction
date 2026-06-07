import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("California House Price Prediction")
st.write("Predict house prices using scikit-learn regression on the California Housing dataset")

# Load model - FIXED PATH AND USE JOBLIB
@st.cache_resource
def load_model():
    model_path = "models/house_price_model.joblib" # Changed this line
    if os.path.exists(model_path):
        return joblib.load(model_path) # Changed to joblib
    else:
        return None

model = load_model()

st.sidebar.header("House Features")
MedInc = st.sidebar.slider("Median Income (10k USD)", 0.5, 15.0, 8.3, 0.1)
HouseAge = st.sidebar.slider("House Age (years)", 1, 52, 28)
AveRooms = st.sidebar.slider("Average Rooms", 2.0, 10.0, 5.2, 0.1)
AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 3.0, 1.1, 0.1)