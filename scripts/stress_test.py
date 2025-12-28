import asyncio
import httpx
import time
import numpy as np
import json
import os
import sys
import random
import argparse

# 1. Setup the Argument Parser
parser = argparse.ArgumentParser(description="Stress test for Turbofan RUL API")

# Environment flag (choices ensure only valid options are picked)
parser.add_argument(
    "--env", 
    choices=["local", "live"], 
    default="local", 
    help="Target environment: 'local' or 'live' (Hugging Face)"
)

# Total requests flag (type=int handles conversion automatically)
parser.add_argument(
    "--total", 
    type=int, 
    default=100, 
    help="Total number of requests to send (default: 100)"
)

# Concurrency flag
parser.add_argument(
    "--concurrent", 
    type=int, 
    default=10, 
    help="Number of concurrent requests (default: 10)"
)

args = parser.parse_args()

# 2. Define URLs and select target
LOCAL_URL = "http://localhost:7860/predict"
LIVE_URL = "https://shuvro77-turbofan-predictive-maintenance.hf.space/predict"
API_URL = LIVE_URL if args.env == "live" else LOCAL_URL

METADATA_PATH = "artifacts/metadata.json"

# Load feature names
with open(METADATA_PATH, 'r') as f:
    meta = json.load(f)
    FEATURE_NAMES = meta['sensor_names']

async def send_request(client, request_id):
    num_sensors = len(FEATURE_NAMES)
    num_cycles = random.randint(1, 100)
    
    raw_data = np.random.rand(num_cycles, num_sensors)
    dict_data = [{FEATURE_NAMES[i]: row[i] for i in range(num_sensors)} for row in raw_data]
    
    payload = {"data_window": dict_data}
    
    start = time.perf_counter()
    try:
        response = await client.post(API_URL, json=payload, timeout=30.0)
        latency = time.perf_counter() - start
        return response.status_code, latency
    except Exception:
        return 500, 0

async def run_stress_test(total_requests, concurrency):
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        tasks = [send_request(client, i) for i in range(total_requests)]
        print(f"🚀 Environment: {args.env.upper()} | Target: {API_URL}")
        print(f"🚀 Stress Test: Total={total_requests}, Concurrency={concurrency}")
        print(f"📊 Each request contains a random window (1-100 cycles)")
        
        results = await asyncio.gather(*tasks)
    
    latencies = [r[1] for r in results if r[0] == 200]
    errors = [r[0] for r in results if r[0] != 200]
    
    if latencies:
        print(f"\n✅ Success: {len(latencies)}/{total_requests}")
        if errors:
            print(f"❌ Errors: {len(errors)}")
        print(f"⏱️ Avg Latency: {np.mean(latencies):.4f}s")
        print(f"📉 P95 Latency: {np.percentile(latencies, 95):.4f}s\n")
    else:
        print("❌ All requests failed. Check Docker logs for error messages.")

if __name__ == "__main__":
    # Use the parsed arguments directly
    asyncio.run(run_stress_test(args.total, args.concurrent))