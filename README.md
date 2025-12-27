# Turbofan Engine RUL Prediction (NASA CMAPSS)

This project uses an **LSTM (Long Short-Term Memory)** network to predict the Remaining Useful Life (RUL) of aircraft engines using the FD001 dataset.

## 🚀 Key Results
* **Final Test MAE:** 9.38
* **Window Size:** 50 Cycles
* **Clipping Value:** 125 (Piecewise RUL)

## 🛠️ Project Structure
* `app/`: FastAPI implementation for real-time inference.
* `artifacts/`: (Local only) Saved models and scalers.
* `notebooks/`: Data exploration and model training.

## 🚦 How to run the API
1. Install requirements: `pip install -r requirements.txt`
2. Run Uvicorn: `uvicorn app.main:app --reload`

## 🧪 Testing the API

To test the prediction engine, you need a 50-cycle window of sensor data formatted as JSON. We have provided a utility script to automate this.

### 1. Generate a Test Payload
Run the following script to pick a random engine from the test set and format its data:
```bash
python3 scripts/get_a_random_valid_input_data_from_test.py
