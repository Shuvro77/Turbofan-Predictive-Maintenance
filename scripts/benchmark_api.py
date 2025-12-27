import requests
import time
import os
import json
import pandas as pd

# Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
api_url = "http://127.0.0.1:8000/predict"
temp_dir = os.path.join(BASE_DIR, 'temporary')

def run_benchmark(num_tests=10):
    print(f"🚀 Starting Benchmark: Sending {num_tests} requests to the API...")
    
    # 1. Get all generated test files from your temporary directory
    json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
    
    if not json_files:
        print("❌ No test data found in /temporary. Run the generator script first!")
        return

    latencies = []
    
    for i in range(min(num_tests, len(json_files))):
        with open(os.path.join(temp_dir, json_files[i]), 'r') as f:
            payload = json.load(f)
        
        # 2. Measure time taken for the request
        start_time = time.time()
        response = requests.post(api_url, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            duration = (end_time - start_time) * 1000 # convert to ms
            latencies.append(duration)
            print(f"Test {i+1}: RUL {response.json()['predicted_remaining_cycles']} | Time: {duration:.2f}ms")
        else:
            print(f"Test {i+1}: Failed with status {response.status_code}")

    # 3. Report Stats
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print("\n--- Benchmark Results ---")
        print(f"Average Latency: {avg_latency:.2f} ms")
        print(f"Min Latency: {min(latencies):.2f} ms")
        print(f"Max Latency: {max(latencies):.2f} ms")

if __name__ == "__main__":
    run_benchmark()