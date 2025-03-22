import pandas as pd
import numpy as np

# ✅ Generate Synthetic Data
np.random.seed(42)
data = {
    "Machine_ID": np.random.randint(1, 1001, size=5000),
    "Vibration_mm_s": np.random.uniform(0.5, 9.5, size=5000),
    "Temperature_C": np.random.uniform(30, 95, size=5000),
    "Pressure_Pa": np.random.uniform(500, 5000, size=5000),
    "Running_Hours": np.random.randint(500, 5000, size=5000),
    "Fault_Type": np.random.choice(["No Fault", "Pressure Leak", "Bearing Failure", "Motor Failure", "Overheating"], size=5000)
}

df = pd.DataFrame(data)

# ✅ Save the Dataset
df.to_csv("synthetic_machine_data.csv", index=False)
print("✅ Synthetic Dataset Created: synthetic_machine_data.csv")
