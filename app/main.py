from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import json

# 1. Initialize FastAPI app
app = FastAPI(title="NASA Turbofan RUL Predictor")

# 2. Load the trained artifacts
# Make sure these files exist in your 'artifacts/' folder
# Use absolute-style paths relative to the project root or use '..'
model = tf.keras.models.load_model('artifacts/lstm_model_v1.keras')
scaler = joblib.load('artifacts/scaler.pkl')

# Define the exact sensor columns we used in training
with open('artifacts/metadata.json', 'r') as f:
    meta = json.load(f)
SENSORS = meta['sensor_names']

# 3. Define the Input Data Schema
class EngineData(BaseModel):
    # A list of 50 dictionaries, each containing the 14 sensors
    data_window: list[dict] 

@app.get("/")
def read_root():
    return {"status": "Model is Live", "MAE": 9.38}

@app.post("/predict")
def predict_rul(input_data: EngineData):
    try:
        # Convert JSON to DataFrame
        df = pd.DataFrame(input_data.data_window)
        
        # Validation: Ensure we have exactly 50 cycles
        if len(df) != 50:
            raise HTTPException(status_code=400, detail="Window must be exactly 50 cycles.")
        
        # 4. Pre-processing
        # Ensure columns are in the correct order
        df = df[SENSORS]

        print("--- SCALER DEBUG INFO ---")
        print(f"Scaler expects these columns: {scaler.feature_names_in_.tolist()}")
        print(f"Your DataFrame has these columns: {df.columns.tolist()}")
        print("-------------------------")
        
        # Scale the data using the loaded scaler
        scaled_data = scaler.transform(df)
        
        # Reshape to (1, 50, 14) for LSTM input
        input_tensor = np.expand_dims(scaled_data, axis=0)
        
        # 5. Inference
        prediction = model.predict(input_tensor)
        rul_result = float(prediction[0][0])
        
        return {
            "predicted_remaining_cycles": round(rul_result, 2),
            "unit": "cycles",
            "status": "Healthy" if rul_result > 30 else "Maintenance Required"
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# To run this: uvicorn main:app --reload
