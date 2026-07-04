from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "Data_Preparation" / "src" / "data.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "Data_Preparation" / "src" / "engineered_data.csv"

data = pd.read_csv(r'Data_Preparation\src\data.csv')

# Create engineered features
# 1. Temperature difference
data["Temperature_Difference"] = data["Process_Temp"] - data["Air_Temp"]

# 2. Machine age score
data["Machine_Age_Score"] = data["Tool_Wear"] * data["Maintenance_Days_Ago"]

# 3. Power consumption
data["Power_Consumption"] = data["Voltage"] * data["Current"]

# 4. Efficiency
data["Efficiency"] = data["Production_Count"] / data["Current"]

# 5. Risk score using temperature, vibration, and tool wear
# Normalize values before combining to keep scale comparable
for col in ["Temperature_Difference", "Vibration", "Tool_Wear"]:
    data[f"{col}_scaled"] = (data[col] - data[col].mean()) / data[col].std()

data["Risk_Score"] = (
    data["Temperature_Difference_scaled"]
    + data["Vibration_scaled"]
    + data["Tool_Wear_scaled"]
)

# Optional: remove temporary scaled columns if not needed
# data = data.drop(columns=[...])

# Save engineered dataset
data.to_csv(r'Data_Preparation\src\engineered_data.csv', index=False)
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "Data_Preparation" / "src" / "engineered_data.csv"

print(f"Engineered data saved to: {OUTPUT_PATH}")
print(data.head())
