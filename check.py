import pickle
import numpy as np

# ✅ Load Model & Scaler
model = pickle.load(open("predictive_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ✅ Test Sample Input (Without Machine ID)
input_data = np.array([[5.0, 50.0, 2500, 2500]])  # ✅ Removed Machine_ID
import pandas as pd

# Convert input_data to DataFrame before scaling
feature_names = ["Vibration_mm_s", "Temperature_C", "Pressure_Pa", "Running_Hours"]
input_df = pd.DataFrame(input_data, columns=feature_names)

# Normalize using the trained scaler
input_data = scaler.transform(input_df)


# ✅ Get Prediction
probabilities = model.predict_proba(input_data)[0]
predicted_index = np.argmax(probabilities)
predicted_fault = label_encoder.inverse_transform([predicted_index])[0]

print(f"⚡ Prediction: {predicted_fault}")
print("🔍 Probability Distribution:")
for idx, prob in enumerate(probabilities):
    fault_name = label_encoder.inverse_transform([idx])[0]
    print(f"{fault_name}: {prob:.2f}")
