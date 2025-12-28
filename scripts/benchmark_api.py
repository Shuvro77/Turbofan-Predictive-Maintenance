import requests
import time
import os
import json
import argparse

parser = argparse.ArgumentParser(description="Benchmark for Turbofan RUL API")
parser.add_argument("--env", choices=["local", "live"], default="local")
parser.add_argument("--num_tests", type=int, default=10)
args = parser.parse_args()

LOCAL_URL = "http://127.0.0.1:7860/predict"
LIVE_URL = "https://shuvro77-turbofan-predictive-maintenance.hf.space/predict"
API_URL = LIVE_URL if args.env == "live" else LOCAL_URL

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
temp_dir = os.path.join(BASE_DIR, 'temporary')

def run_benchmark(num_tests):
    print(f"🚀 Environment: {args.env.upper()} | Target: {API_URL}")
    
    if not os.path.exists(temp_dir):
        print(f"❌ Directory not found: {temp_dir}")
        return

    json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
    if not json_files:
        print("❌ No test data found. Run the generator script first!")
        return

    latencies = []
    
    for i in range(min(num_tests, len(json_files))):
        with open(os.path.join(temp_dir, json_files[i]), 'r') as f:
            payload = json.load(f)
        
        start_time = time.time()
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            end_time = time.time()
            
            if response.status_code == 200:
                duration = (end_time - start_time) * 1000
                latencies.append(duration)
                
                # Check for multiple possible key names
                res_data = response.json()
                rul = res_data.get('prediction', res_data.get('predicted_remaining_cycles', "Unknown"))
                
                print(f"Test {i+1}: RUL {rul} | Time: {duration:.2f}ms")
            else:
                print(f"Test {i+1}: Failed (Status {response.status_code})")
        except Exception as e:
            print(f"Test {i+1}: Error - {str(e)}")

    if latencies:
        print("\n--- Benchmark Results ---")
        print(f"Average Latency: {sum(latencies) / len(latencies):.2f} ms")
        print(f"Min Latency: {min(latencies):.2f} ms")
        print(f"Max Latency: {max(latencies):.2f} ms")

if __name__ == "__main__":
    run_benchmark(args.num_tests)