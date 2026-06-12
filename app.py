import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.linear_model import LinearRegression

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="House Price AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INITIALIZE ALL VARIABLES ---
predicted_price = 0
model = None
df = pd.DataFrame()

# --- HEADER ---
st.title("🏠 House Price Predictor")
st.caption("Machine Learning Model • King County Dataset • Built with XGBoost + Plotly")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("kc_house_data.csv")
        return data
    except FileNotFoundError:
        st.error("❌ Missing kc_house_data.csv. Upload it to GitHub.")
        st.stop()

df = load_data()

# --- LOAD OR TRAIN MODEL ---
@st.cache_resource
def load_model():
    MODEL_PATH = "model.pkl"
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.warning(f"Model file corrupted: {e}. Training new model.")

    # Fallback: train simple model if model.pkl missing
    st.info("Training model live... this takes 5 seconds")
    X = df[['sqft_living', 'bedrooms', 'bathrooms', 'yr_built']]
    y = df['price']
    new_model = LinearRegression().fit(X, y)
    return new_model

model = load_model()

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Enter House Details")
    sqft = st.number_input("Square Feet", min_value=300, max_value=15000, value=2000, step=50)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1.0, 5.0, 2.0, 0.5)
    yr_built = st.number_input("Year Built", 1900, 2026, 2000)

    st.markdown("---")
    st.caption("Model: Linear Regression")
    st.caption(f"Trained on {len(df):,} houses")

# --- PREDICTION ---
try:
    features = [[sqft, bedrooms, bathrooms, yr_built]]
    predicted_price = model.predict(features)[0]
except Exception as e:
    st.error(f"Prediction Error: {e}")
    predicted_price = 0

# --- MAIN DASHBOARD ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Price", f"${predicted_price:,.0f}" if predicted_price > 0 else "Error")
with col2:
    st.metric("Price per Sq Ft", f"${predicted_price/sqft:.0f}" if predicted_price > 0 else "N/A")
with col3:
    avg_price = df['price'].mean()
    diff = ((predicted_price - avg_price) / avg_price * 100) if predicted_price > 0 else 0
    st.metric("vs Market Avg", f"{diff:+.1f}%", delta=f"{diff:+.1f}%")

st.markdown("---")

# --- CHARTS ---
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_hist = px.histogram(
        df, x="price", nbins=60,
        title="Market Price Distribution",
        labels={'price': 'Price (USD)', 'count': 'Number of Houses'}
    )
    if predicted_price > 0:
        fig_hist.add_vline(
            x=predicted_price, line_color="red", line_width=3, line_dash="dash",
            annotation_text="Your Prediction", annotation_position="top right"
        )
    fig_hist.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

with chart_col2:
    fig_scatter = px.scatter(
        df.sample(2000), x="sqft_living", y="price",
        opacity=0.4, title="Price vs Square Feet",
        labels={'sqft_living': 'Square Feet', 'price': 'Price (USD)'},
        trendline="ols", trendline_color_override="blue"
    )
    if predicted_price > 0:
        fig_scatter.add_trace(
            go.Scatter(x=[sqft], y=[predicted_price], mode='markers',
                      marker=dict(size=15, color='red', symbol='star'),
                      name='Your House')
        )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- FEATURE IMPORTANCE ---
if hasattr(model, 'feature_importances_'):
    st.subheader("What Drives Price?")
    importance_df = pd.DataFrame({
        'feature': ['Sq Ft', 'Bedrooms', 'Bathrooms', 'Year Built'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    fig_importance = px.bar(
        importance_df, x='importance', y='feature', orientation='h',
        title="Feature Importance from Model"
    )
    fig_importance.update_layout(height=300)
    st.plotly_chart(fig_importance, use_container_width=True)
else:
    st.info("Feature importance not available for Linear Regression. Switch to XGBoost to see this.")

# --- FOOTER ---
st.markdown("---")
st.caption("Built by Benedict Omondi • Data: King County, WA • Updated 2026")