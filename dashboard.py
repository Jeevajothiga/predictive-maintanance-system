import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ✅ Load Model & Encoders
model = pickle.load(open("predictive_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# 🎨 Streamlit UI
st.title("🚀 AI-Powered Predictive Maintenance Dashboard")
st.write("Predict machine failures based on sensor data!")

# ✅ Create or Load Previous Predictions
csv_file = "prediction_history.csv"
if os.path.exists(csv_file):
    history_data = pd.read_csv(csv_file)
else:
    history_data = pd.DataFrame(columns=["Vibration_mm_s", "Temperature_C", "Pressure_Pa", "Running_Hours", "Predicted_Fault"])

# 🔹 Input Features (Without Machine ID)
vibration = st.number_input("Vibration (mm/s)", min_value=0.1, max_value=10.0, value=5.0)
temperature = st.number_input("Temperature (°C)", min_value=20.0, max_value=100.0, value=50.0)
pressure = st.number_input("Pressure (Pa)", min_value=500, max_value=5000, value=2500)
running_hours = st.number_input("Running Hours", min_value=100, max_value=5000, value=2500)

# 🟢 Predict Button
if st.button("🔍 Predict Machine Fault"):
    input_data = np.array([[vibration, temperature, pressure, running_hours]])  # ✅ Removed Machine ID
    input_data = scaler.transform(input_data)  # Normalize

    # ✅ Get Prediction
    probabilities = model.predict_proba(input_data)[0]
    predicted_index = np.argmax(probabilities)
    predicted_fault = label_encoder.inverse_transform([predicted_index])[0]

    st.success(f"⚡ Prediction: {predicted_fault}")

    # ✅ Show Probability Distribution
    for idx, prob in enumerate(probabilities):
        fault_name = label_encoder.inverse_transform([idx])[0]
        st.write(f"{fault_name}: {prob:.2f}")

    # ✅ Store the Prediction
    new_data = pd.DataFrame({
        "Vibration_mm_s": [vibration],
        "Temperature_C": [temperature],
        "Pressure_Pa": [pressure],
        "Running_Hours": [running_hours],
        "Predicted_Fault": [predicted_fault]
    })
    
    history_data = pd.concat([history_data, new_data], ignore_index=True)
    history_data.to_csv(csv_file, index=False)

# ✅ Show Updated Prediction History
st.write("📊 **Prediction History:**")
st.dataframe(history_data.tail(10))

# ✅ Display Fault Type Distribution
if not history_data.empty:
    st.write("📊 **Fault Type Distribution:**")
    fig, ax = plt.subplots()
    sns.countplot(x=history_data["Predicted_Fault"], palette="coolwarm", ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)
