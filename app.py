import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("California House Price Prediction")
st.write("Predict house prices using scikit-learn regression on the California Housing dataset")

# Load model - adjust path if your model is saved elsewhere
@st.cache_resource
def load_model():
    model_path = "models/model.pkl" # Change this to match your train.py output
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

model = load_model()

st.sidebar.header("House Features")
MedInc = st.sidebar.slider("Median Income (10k USD)", 0.5, 15.0, 8.3, 0.1)
HouseAge = st.sidebar.slider("House Age (years)", 1, 52, 28)
AveRooms = st.sidebar.slider("Average Rooms", 2.0, 10.0, 5.2, 0.1)
AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 3.0, 1.1, 0.1)
Population = st.sidebar.number_input("Block Population", 100, 35000, 1425)
AveOccup = st.sidebar.slider("Average Occupancy", 1.0, 6.0, 3.0, 0.1)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 35.6, 0.1)
Longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -119.5, 0.1)

if st.sidebar.button("Predict Price"):
    if model is None:
        st.error("Model file not found. Run train.py first to create models/model.pkl")
    else:
        features = [[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]]
        prediction = model.predict(features)[0] * 100000 # Dataset is in 100k units

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted House Value", f"${prediction:,.0f}")
        with col2:
            st.metric("Price per Room", f"${prediction/AveRooms:,.0f}")

        st.success("Prediction complete")
        st.info("Model trained on California Housing dataset from 1990")
else:
    st.info("Enter house details in sidebar and click Predict Price")

st.markdown("---")
st.caption("Model: Linear/Ridge Regression | Dataset: California Housing | Features: 8")