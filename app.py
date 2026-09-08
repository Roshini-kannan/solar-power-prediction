import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("solar_power_prediction_model.pkl")

model = load_model()

st.title("☀️ Solar Power Prediction")
st.write("Enter the environmental conditions to predict solar power output.")

st.divider()

temperature = st.number_input(
    "Temperature",
    value=25.0
)

humidity = st.number_input(
    "Humidity",
    value=50.0
)

solar_irradiance = st.number_input(
    "Solar Irradiance",
    value=500.0
)

wind_speed = st.number_input(
    "Wind Speed",
    value=5.0
)

if st.button("Predict Solar Power", type="primary"):

    input_data = pd.DataFrame(
        [[
            temperature,
            humidity,
            solar_irradiance,
            wind_speed
        ]],
        columns=[
            "temperature",
            "humidity",
            "solar_irradiance",
            "wind_speed"
        ]
    )

    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    st.success(
        f"☀️ Predicted Solar Power Output: {prediction:.2f}"
    )
