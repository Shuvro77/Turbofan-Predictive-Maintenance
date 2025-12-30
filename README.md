---
title: Turbofan Predictive Maintenance
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
###  Deployment Info
- **Deployment Status:** ![Deployment Status](https://github.com/Shuvro77/Turbofan-Predictive-Maintenance/actions/workflows/deploy.yaml/badge.svg)
- **Last Updated:** Refer to the [GitHub Actions History](https://github.com/Shuvro77/Turbofan-Predictive-Maintenance/actions) for the latest deployment timestamp.
- **Live Demo:** [Hugging Face Space API](https://huggingface.co/spaces/Shuvro77/Turbofan-Predictive-Maintenance)

> ** Automated Deployment Note:** This project is configured with **CI/CD**. Any changes committed and pushed to the `main` branch will automatically trigger a build and deploy the updated container to the Hugging Face Space.

# Turbofan Engine RUL Prediction (NASA CMAPSS)

This project uses an **LSTM (Long Short-Term Memory)** network to predict the Remaining Useful Life (RUL) of aircraft engines using the FD001 dataset.

## Quick Navigation

| Section | Description |
| :--- | :--- |
| [Machine Learning Workflow](#machine-learning-workflow) | Details on data engineering, feature selection, and LSTM architecture. |
| [Project Structure](#project-structure) | Overview of the repository organization and file locations. |
| [How to Run the API](#how-to-run-the-api) | Installation steps and local execution instructions. |
| [Testing the API](#testing-the-api) | Guide for generating test payloads and verifying endpoints. |
| [Live API Access](#live-api-access) | Access points for the production Hugging Face Space. |
| [Performance Benchmarking](#performance-benchmarking) | Latency analysis across different hosting environments. |
| [Docker Optimization](#docker-optimization--deployment) | Technical breakdown of image size reduction techniques. |

---


## Key Results
* **Final Test MAE:** 9.38
* **Window Size:** 50 Cycles
* **Clipping Value:** 125 (Piecewise RUL)
---
## Machine Learning Workflow

This project follows a structured MLOps pipeline to predict the Remaining Useful Life (RUL) of aircraft engines using the NASA C-MAPSS dataset.

### 1. Data Engineering & Preprocessing
- **Dataset:** NASA CMAPSS (FD001), involving 21 sensors and 3 operational settings.
- **Normalization:** Min-Max scaling was applied to sensor values to ensure they fall within the range [0, 1], preventing sensors with larger magnitudes from dominating the loss function.
- **Labeling:** The target variable (RUL) was clipped at 125 cycles. This "Piecewise Linear RUL" strategy acknowledges that engines do not show signs of degradation until they reach a certain wear threshold.
![RUL Distribution](reports/figures/rul_distribution.png)
- **RemovingZeroVarianceFeatures:** The features with almost zero variance/dead features have been dropped. **['setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19']**
![RUL Distribution](reports/figures/columns_with_zero_variance.png)
- **DroppingLowCorrelatedFeatures** Low correlated features (setting_1, setting_2) have been removed.  ![RUL Distribution](reports/figures/low_correlated_features.png)
- **RemovedRedundantFeatures:** To remove multicolinear feature such as `s_14` was removed with (Correlation > 0.95).
- **ValidfeatureSets:** 14 features used for training. ['s_11', 's_12', 's_13', 's_15', 's_17', 's_2', 's_20', 's_21', 's_3', 's_4', 's_6', 's_7', 's_8', 's_9'] 
- **Sliding Window:** To capture temporal dependencies, data was reshaped into 3D sequences (Samples, Time Steps, Features) with a window size of **50 cycles**.

### 2. Model Architecture (LSTM)
The model uses a **Long Short-Term Memory (LSTM)** network, which is ideal for time-series forecasting due to its ability to remember long-term dependencies in sensor patterns.
###  Model Summary
The following table outlines the layer structure, output shapes, and parameter counts of the LSTM model used for RUL prediction:

```
Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ lstm_4 (LSTM)                   │ (None, 50, 64)         │        20,224 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_4           │ (None, 50, 64)         │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_4 (Dropout)             │ (None, 50, 64)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_5 (LSTM)                   │ (None, 32)             │        12,416 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_5           │ (None, 32)             │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_5 (Dropout)             │ (None, 32)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (Dense)                 │ (None, 16)             │           528 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_5 (Dense)                 │ (None, 1)              │            17 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 33,569 (131.13 KB)
 Trainable params: 33,377 (130.38 KB)
 Non-trainable params: 192 (768.00 B)
```
### 3. Training & Evaluation
- **Optimizer:** Adam optimizer with a Mean Squared Error (MSE) loss function.
- **Early Stopping:** Implemented to monitor validation loss and halt training when performance plateaued, ensuring the model generalizes well to unseen engines.
- **Metrics:** - **MAE (Mean Absolute Error):** Measures the average magnitude of error in cycles.
  - **RMSE (Root Mean Square Error):** Penalizes larger errors, critical for safety-first maintenance. ![Training loss](reports/figures/training_loss.png)
  

### 4. Model Verification
Before deployment, the model was verified using:
- **Test Set Performance:** Evaluated against the ground truth RUL values provided in the CMAPSS dataset.
- **Visual Analysis:** Comparison plots between Predicted RUL vs. Actual RUL for various engine IDs to verify the trend of degradation. ![Final evaluation](reports/figures/final_evaluation_actual_vs_predicted.png)
- **Error Analysis:** The residual distribution is centered around zero, confirming that our model's errors are normally distributed and unbiased. ![Residual analysis](reports/figures/residual_distribution.png)
- **Stress Testing:** Benchmarking the inference latency to ensure the model can handle real-time sensor streams within the FastAPI container.
---
##  Project Structure
* `app/`: FastAPI implementation for real-time inference.
* `artifacts/`: (Local only) Saved models and scalers.
* `notebooks/`: Data exploration and model training.
* `reports/figures`: Figures plotted during training and evaluation
* `scripts`: Scripts needed for various helps

##  How to run the API
1. Install requirements: `pip install -r requirements.txt`
2. Run Uvicorn: `uvicorn app.main:app --reload`

---
## Testing the API

The API is designed to handle time-series windows of varying lengths. While the LSTM model requires exactly 50 cycles, the system implements **Zero-Padding** to support engines with shorter histories.

### 1. Generate a Test Payload
Run the utility script to generate a random test case. It will automatically pick a random engine and a window size between 10 and 50 cycles:
```
python3 scripts/get_a_random_valid_input_data_from_test.py
```

### 2. Send a Prediction Request
You can use the built-in Swagger UI at http://127.0.0.1:8000/docs or use curl:
```
curl -X 'POST' '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
     -H 'Content-Type: application/json' \
     -d @temporary/random_data_unit_X.json
```

### 3. Response Format
```
{
  "predicted_remaining_cycles": 127.03,
  "input_cycles_received": 32,
  "padding_applied": true
}
```
---
## Live API Access

The model is professionally hosted on **Hugging Face Spaces** using a Dockerized FastAPI backend. You can interact with the API directly via the web interface or programmatically.

**API Base URL:** `https://shuvro77-turbofan-predictive-maintenance.hf.space`

### **1. Interactive Documentation**
Explore the endpoints and test the model in your browser via Swagger UI:
 [View Interactive Docs](https://shuvro77-turbofan-predictive-maintenance.hf.space/docs)

---
## Performance Benchmarking

To ensure the API is production-ready, we track latency metrics. Deep learning models often experience a "cold start" on the first request as the computation graph initializes.

### 1. Latency Benchmarking
Ensure the server is running, then execute:
#### Local Environment
- **API Documentation:** Visit `http://localhost:7860/docs`
- **Check Logs:** `sudo docker logs -f turbofan-container`
- **Run Benchmark:**
  ```
  python3 scripts/benchmark_api.py --env local
  ```
#### Live Environment (Hugging Face)
- **API Documentation:** Visit `https://shuvro77-turbofan-predictive-maintenance.hf.space/docs`
- **Run Benchmark:**
  ```
  python3 scripts/benchmark_api.py --env live
  ```
#### Benchmark Comparison Results
| Metric | Local Docker (8-Core CPU) | Live HF Space (Shared CPU) |
|:-------------------|:-------------|:------------|
| Avg Latency                 | 89.21 ms         | 2312.13 ms       |
| Min Latency                 | 83.86 ms        | 1562.90 ms       | 
| Max Latency                 | 92.33 ms        | 2643.89 ms       | 
---



### 2. High-Concurrency Stress Testing

To evaluate the operational limits of the API, stress tests were conducted comparing a **Local Environment** (Dedicated CPU) vs. **Hugging Face Spaces** (Shared Free-Tier CPU).

#### Local Environment (Development)
* **Hardware:** Local Host
* **Target:** `http://localhost:7860`

| Concurrency (Users) | Success Rate | Avg Latency | P95 Latency |
|:-------------------|:-------------|:------------|:------------|
| 10                 | 100%         | 3.73s       | 6.10s       |
| 20                 | 100%         | 2.36s       | 3.65s       |
| 50                 | 100%         | 2.28s       | 3.43s       |
| 80                 | 100%         | 2.50s       | 3.43s       |

#### Live Production (Hugging Face)
* **Hardware:** 2 vCPU, 16GB RAM (Shared)
* **Target:** `https://shuvro77-turbofan-predictive-maintenance.hf.space`

| Concurrency (Users) | Success Rate | Avg Latency | P95 Latency |
|:-------------------|:-------------|:------------|:------------|
| 10                 | 100%         | 21.64s      | 40.70s      |
| 20                 | 100%         | 19.26s      | 30.24s      |
| 40                 | 98%          | 26.07s      | 41.81s      |
| 80                 | 37%          | 20.07s      | 43.58s      |

#### **Key Observations**
- **Infrastructure Limits:** The transition from local to shared cloud infrastructure resulted in a ~10x increase in latency, typical for computationally intensive LSTM models running on shared vCPUs.
- **Stability Threshold:** The production API remains highly stable up to **20 concurrent users**. Beyond **40 concurrent users**, we observe request queuing and a drop in success rates due to CPU throttling.
- **Reliability:** The 16GB RAM allocation on Hugging Face ensures the model remains loaded without Out-Of-Memory (OOM) errors even under heavy concurrent load.
- **Key Finding:** The system is highly stable under load, maintaining a 100% success rate even at 80 concurrent connections. The latency remains consistent, demonstrating that the FastAPI + Uvicorn setup efficiently queues and processes LSTM inference tasks.

#### **How to Run Stress Tests**
Ensure the Docker container is running, then use:
```
# Usage: stress_test.py [-h] [--env {local,live}] [--total TOTAL] [--concurrent CONCURRENT]

Local:
python3 scripts/stress_test.py --env local --total 100 --concurrent 20

Live:
python3 scripts/stress_test.py --env live --total 100 --concurrent 10
```

---

## Docker Optimization & Deployment

The Docker image has been optimized for production environments, focusing on reducing the footprint for cloud hosting (e.g., Render, Hugging Face Spaces).

### **Running with Docker**

This project is fully containerized. Docker ensures the API runs in an environment with the exact versions of TensorFlow and Python required for the LSTM model.

#### 1. Build the Image
From the project root, run:
```
sudo docker build -t turbofan-rul-api:latest .
```

#### 2. Run the Container
Start the API in detached mode, mapping port 8000:
```
sudo docker run -d -p 8000:8000 --name turbofan-container turbofan-rul-api:latest
```
#### 3. Verify the Deployment
API Documentation: Visit http://localhost:8000/docs

Check Logs: `sudo docker logs -f turbofan-container`

Run Benchmark: Ensure the container is running, then execute the local benchmark script:
```
python3 scripts/benchmark_api.py
```
Benchmark output in docker
```
(venv) shuvro@shuvro:~/Desktop/Turbofan_RUL_Project$ python3 scripts/benchmark_api.py
🚀 Starting Benchmark: Sending 10 requests to the API...
Test 1: RUL 127.03 | Time: 83.86ms
Test 2: RUL 114.51 | Time: 88.16ms
Test 3: RUL 0.67 | Time: 91.02ms
Test 4: RUL 0.67 | Time: 90.71ms
Test 5: RUL 84.33 | Time: 92.33ms

--- Benchmark Results ---
Average Latency: 89.21 ms
Min Latency: 83.86 ms
Max Latency: 92.33 ms
```

#### 4. Stopping the Project
To stop and remove the container:
```
sudo docker stop turbofan-container
sudo docker rm turbofan-container
```
### **Optimization Techniques Applied:**
1.  **Multi-Stage Builds:** Separated the build environment (compilers/pip cache) from the runtime environment.
2.  **Library Selection:** Switched to `tensorflow-cpu` to remove ~1.5GB of unnecessary GPU/CUDA binaries.
3.  **Venv Pruning:** Manually removed `__pycache__` and `.pyc` files from the virtual environment during the build process.
4.  **Base Image:** Utilized `python:3.12-slim` to minimize the underlying OS layer.

### **Image Evolution:**
| Version | Image Size | Notes |
| :--- | :--- | :--- |
| Initial Build | 5.34 GB | Included local `venv` and full TensorFlow |
| Optimized v1 | 2.57 GB | Added `.dockerignore` and multi-stage build |
| **Current (Prod)**| **1.77 GB** | Switched to `tensorflow-cpu` + venv pruning |

### **How to Build the Optimized Image:**
```
sudo docker build -t turbofan-rul-api:prod .
```