from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "rajasthan_flood_weather_merged_1986_2022.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "ridge_flood_risk_model.joblib"


# Features used by the model
FEATURES = [
    "avg_annual_rainfall_mm",
    "avg_temperature_c",
    "avg_humidity_percent",
    "avg_wind_speed_mps",
    "avg_pressure_kpa",
    "max_daily_rainfall_mm",
]

TARGET = "flood_affected_percent"


# Load dataset
df = pd.read_csv(DATA_PATH)

# Keep only required columns and remove missing values
model_df = df[FEATURES + [TARGET]].dropna()

X = model_df[FEATURES]
y = model_df[TARGET]


# Train Ridge Regression model
model = Ridge(alpha=1.0)
model.fit(X, y)


# Create models directory if needed
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Save model + feature information
model_package = {
    "model": model,
    "features": FEATURES,
    "target": TARGET,
}


joblib.dump(model_package, MODEL_PATH)


print("========================================")
print("Aegis Ridge Model Training Complete")
print("========================================")
print(f"Training records : {len(model_df)}")
print(f"Features         : {len(FEATURES)}")
print(f"Target           : {TARGET}")
print(f"Model saved to   : {MODEL_PATH}")
print("========================================")
