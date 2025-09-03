import streamlit as st
import numpy as np
import joblib

# Inject custom CSS for styling
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background-color: #f0f8ff;
    }
    /* Title color */
    h1 {
        color: #4CAF50;
    }
    /* Button style */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    /* Text color */
    .css-1d391kg p {
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load pre-trained model and scaler
model = joblib.load('stress_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🌸🌿 Stress Detector App 🧠💡")


st.write("Enter your measurements below:")

# User inputs for features
humidity = st.slider("Humidity (%)", 0, 100, 50)
temperature = st.slider("Temperature (°F)", 50, 110, 75)
step_count = st.number_input("Step count", min_value=0, max_value=10000, value=1000)

heart_rate = st.slider("Heart Rate (bpm)", 40, 180, 70)
respiration_rate = st.slider("Respiration Rate (breaths per min)", 10, 40, 16)
sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
systolic = st.number_input("Systolic Blood Pressure", 80, 200, 120)
diastolic = st.number_input("Diastolic Blood Pressure", 50, 130, 80)
mood = st.slider("Mood Level (1-10)", 1, 10, 5)
caffeine_intake = st.slider("Caffeine Intake (mg/day)", 0, 600, 100)

if st.button("Predict Stress Level"):
    # Prepare input array with all features
    input_features = np.array([[
        humidity, temperature, step_count,
        heart_rate, respiration_rate, sleep_hours,
        systolic, diastolic, mood, caffeine_intake
    ]])

    # Scale the input features
    scaled_features = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(scaled_features)

    # Map prediction to meaningful stress level labels
    stress_levels = {0: "Low Stress", 1: "Medium Stress", 2: "High Stress"}
    result = stress_levels.get(prediction[0], "Unknown")

    st.success(f"Predicted Stress Level: {result}")
