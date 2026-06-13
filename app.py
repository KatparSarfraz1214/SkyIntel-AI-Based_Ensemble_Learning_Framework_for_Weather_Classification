import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import date
import plotly.express as px

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI Weather Intelligence System",
    page_icon="",
    layout="wide"
)

# ----------------------------
# CSS (Professional UI Updated)
# ----------------------------
st.markdown("""
<style>

body {
    background-color: #0f172a;
}

/* Main app background */
.main {
    background-color: #f8fafc;
    font-size: 18px;
}

/* Title */
.title {
    font-size: 53px;
    font-weight: 900;
    text-align: center;
    color: white;
    margin-top: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 30px;
    margin-bottom: 25px;
}

/* Big result box */
.big-result {
    padding: 23px;
    border-radius: 15px;
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white;
    font-size: 34px;
    font-weight: bold;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("weather_rf_model.pkl", "rb"))
    encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return model, encoder

model, le = load_model()

# ----------------------------
# HEADER
# ----------------------------
st.markdown("<div class='title'>SkyIntel</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Based Ensemble Learning Framework for Weather Classification</div>", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("Weather Control Panel")

selected_date = st.sidebar.date_input("Select Date", date.today())

precipitation = st.sidebar.slider("Precipitation", 0.0, 100.0, 0.0)
temp_max = st.sidebar.slider("Max Temperature (°C)", -10.0, 50.0, 25.0)
temp_min = st.sidebar.slider("Min Temperature (°C)", -20.0, 40.0, 15.0)
wind = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, 5.0)

year = selected_date.year
month = selected_date.month
day = selected_date.day
dayofweek = selected_date.weekday()

# ----------------------------
# INPUT FRAME
# ----------------------------
input_df = pd.DataFrame({
    "precipitation": [precipitation],
    "temp_max": [temp_max],
    "temp_min": [temp_min],
    "wind": [wind],
    "year": [year],
    "month": [month],
    "day": [day],
    "dayofweek": [dayofweek]
})

# ----------------------------
# TABS (Cleaned)
# ----------------------------
tab1, tab2 = st.tabs(["Prediction", "Insights"])

# ============================
# TAB 1 - PREDICTION
# ============================
with tab1:

    # White metric boxes and columns removed from here to streamline layout

    if st.button("Predict Weather Now"):

        pred = model.predict(input_df)
        weather = le.inverse_transform(pred)[0]

        probs = model.predict_proba(input_df)[0]
        confidence = np.max(probs) * 100

        # BIG RESULT BOX
        st.markdown(
            f"<div class='big-result'>{weather.upper()}</div>",
            unsafe_allow_html=True
        )

        st.success(f"Model Confidence: {confidence:.2f}%")

        # Probability Chart
        prob_df = pd.DataFrame({
            "Weather": le.classes_,
            "Probability": probs * 100
        })

        fig = px.bar(
            prob_df,
            x="Weather",
            y="Probability",
            text="Probability"
        )

        st.plotly_chart(fig, use_container_width=True)

# ============================
# TAB 2 - INSIGHTS
# ============================
with tab2:

    st.subheader("Feature Importance Analysis")

    importance_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance")

    fig2 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Input Summary")

    st.dataframe(input_df, use_container_width=True)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("Built with Streamlit + Scikit-Learn + Random Forest | AI Weather Intelligence System")