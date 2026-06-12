import streamlit as st
import plotly.express as px
import pandas as pd

# Initialize everything - prevents NameError
fig_hist = None
fig_scatter = None
predicted_price = 0
df = pd.DataFrame() # your data

st.title("House Price Predictor")

# User inputs
sqft = st.number_input("Square Feet", 500, 10000, 2000)
bedrooms = st.slider("Bedrooms", 1, 6, 3)

# Prediction logic wrapped in try/except
try:
    predicted_price = model.predict([[sqft, bedrooms]])[0] # your model
    st.metric("Predicted Price", f"${predicted_price:,.0f}")
except Exception as e:
    st.error("Prediction failed. Check inputs.")
    st.stop()

# Only draw charts if we have data + prediction
if not df.empty and predicted_price > 0:
    # Chart 1: Price distribution
    fig_hist = px.histogram(df, x="price", nbins=50,
                           title="Market Price Distribution")
    fig_hist.add_vline(x=predicted_price, line_dash="dash",
                       line_color="red", annotation_text="Your Prediction")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Chart 2: Sqft vs Price
    fig_scatter = px.scatter(df, x="sqft", y="price",
                            trendline="ols", title="Price vs Square Feet")
    fig_scatter.add_scatter(x=[sqft], y=[predicted_price],
                           mode='markers', marker=dict(size=15, color='red'),
                           name='Your House')
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Upload data to see market charts")