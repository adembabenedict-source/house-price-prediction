import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="House Price AI", page_icon="🏠", layout="wide")

# --- Initialize session state ---
if 'prediction' not in st.session_state:
    st.session_state.prediction = 0.0
if 'avg_price' not in st.session_state:
    st.session_state.avg_price = 0.0
if 'r2' not in st.session_state:
    st.session_state.r2 = 0.0
if 'mae' not in st.session_state:
    st.session_state.mae = 0.0
if 'user_input_df' not in st.session_state:
    st.session_state.user_input_df = pd.DataFrame()

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("kc_house_data.csv", encoding='utf-8', on_bad_lines='skip', low_memory=False)
        return data
    except FileNotFoundError:
        st.error("❌ Missing kc_house_data.csv. Upload it to GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"❌ CSV Error: {e}")
        st.stop()

@st.cache_resource
def train_model(df):
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                'sqft_basement', 'yr_built', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

    X = df[features]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, scaler, r2, mae, features

# --- Load Data ---
df = load_data()
model, scaler, r2, mae, features = train_model(df)

# --- UI ---
st.title("🏠 House Price Predictor")
st.markdown("Machine Learning Model • King County Dataset • Built with XGBoost + Plotly")

# --- Sidebar Inputs ---
st.sidebar.header("Enter House Details")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 8, 2, 0.5)
sqft_living = st.sidebar.number_input("Living Area (sqft)", 500, 10000, 2000, 50)
sqft_lot = st.sidebar.number_input("Lot Size (sqft)", 500, 500000, 7000, 100)
floors = st.sidebar.slider("Floors", 1.0, 3.5, 1.0, 0.5)
waterfront = st.sidebar.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x else "No")
view = st.sidebar.slider("View Rating", 0, 4, 0)
condition = st.sidebar.slider("Condition", 1, 5, 3)
grade = st.sidebar.slider("Grade", 1, 13, 7)
sqft_above = st.sidebar.number_input("Above Ground (sqft)", 500, 10000, 1500, 50)
sqft_basement = st.sidebar.number_input("Basement (sqft)", 0, 5000, 500, 50)
yr_built = st.sidebar.number_input("Year Built", 1900, 2025, 1990, 1)
lat = st.sidebar.number_input("Latitude", 47.0, 48.0, 47.55, format="%.4f")
long = st.sidebar.number_input("Longitude", -122.5, -121.5, -122.25, format="%.4f")
sqft_living15 = st.sidebar.number_input("Living Area 2015 (sqft)", 500, 10000, 2000, 50)
sqft_lot15 = st.sidebar.number_input("Lot Size 2015 (sqft)", 500, 500000, 7000, 100)

if st.sidebar.button("Predict Price", type="primary"):
    user_input = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
                           view, condition, grade, sqft_above, sqft_basement, yr_built,
                           lat, long, sqft_living15, sqft_lot15]])

    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]

    st.session_state.prediction = prediction
    st.session_state.avg_price = df['price'].mean()
    st.session_state.r2 = r2
    st.session_state.mae = mae
    st.session_state.user_input_df = pd.DataFrame(user_input, columns=features)

# --- Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Price", f"${st.session_state.prediction:,.0f}")
price_per_sqft = st.session_state.prediction / sqft_living if sqft_living > 0 else 0
col2.metric("Price per Sq Ft", f"${price_per_sqft:.0f}")
vs_avg = ((st.session_state.prediction - st.session_state.avg_price) / st.session_state.avg_price * 100) if st.session_state.avg_price > 0 else 0
col3.metric("vs Market Avg", f"{vs_avg:+.1f}%")

st.markdown("---")

# --- Charts ---
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Market Price Distribution")
    fig_hist = px.histogram(df, x="price", nbins=50, title="King County House Prices")
    fig_hist.add_vline(x=st.session_state.prediction, line_dash="dash", line_color="red",
                      annotation_text="Your Prediction")
    st.plotly_chart(fig_hist, use_container_width=True)

with chart_col2:
    st.subheader("Price vs Square Feet")
    sample_df = df.sample(n=min(2000, len(df)), random_state=42)
    fig_scatter = px.scatter(sample_df, x="sqft_living", y="price",
                            trendline="ols", trendline_color_override="blue",
                            title="Size vs Price")

    if not st.session_state.user_input_df.empty:
        fig_scatter.add_trace(go.Scatter(
            x=st.session_state.user_input_df['sqft_living'],
            y=[st.session_state.prediction],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Your House'
        ))

    st.plotly_chart(fig_scatter, use_container_width=True)

# --- Model Performance ---
st.markdown("---")
st.subheader("Model Performance")
perf_col1, perf_col2 = st.columns(2)
perf_col1.metric("R² Score", f"{st.session_state.r2:.3f}")
perf_col2.metric("Mean Absolute Error", f"${st.session_state.mae:,.0f}")

st.caption("Built by Benedict Omondi • Data: King County, WA • Updated 2026")