import asyncio
import httpx
import time
import numpy as np
import json
import os
import sys
import random

# Configuration
API_URL = "http://localhost:8000/predict"
METADATA_PATH = "artifacts/metadata.json"

# Load feature names
with open(METADATA_PATH, 'r') as f:
    meta = json.load(f)
    FEATURE_NAMES = meta['sensor_names']

async def send_request(client, request_id):
    num_sensors = len(FEATURE_NAMES)
    
    # --- RANDOM CYCLES LOGIC ---
    # Generate a random sequence length between 1 and 100
    num_cycles = random.randint(1, 100)
    
    raw_data = np.random.rand(num_cycles, num_sensors)
    dict_data = [{FEATURE_NAMES[i]: row[i] for i in range(num_sensors)} for row in raw_data]
    
    payload = {"data_window": dict_data}
    
    start = time.perf_counter()
    try:
        response = await client.post(API_URL, json=payload, timeout=30.0)
        latency = time.perf_counter() - start
        
        # Log failure details for the first failure encountered
        if response.status_code != 200:
            return response.status_code, 0
            
        return response.status_code, latency
    except Exception:
        return 500, 0

async def run_stress_test(total_requests, concurrency):
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        tasks = [send_request(client, i) for i in range(total_requests)]
        print(f"🚀 Random Stress Test: Total={total_requests}, Concurrency={concurrency}")
        print(f"📊 Each request contains a random window (1-100 cycles)")
        
        results = await asyncio.gather(*tasks)
    
    latencies = [r[1] for r in results if r[0] == 200]
    errors = [r[0] for r in results if r[0] != 200]
    
    if latencies:
        print(f"\n✅ Success: {len(latencies)}/{total_requests}")
        if errors:
            print(f"❌ Errors: {len(errors)} (Check if model handles > 50 cycles)")
        print(f"⏱️ Avg Latency: {np.mean(latencies):.4f}s")
        print(f"📉 P95 Latency: {np.percentile(latencies, 95):.4f}s\n")
    else:
        print("❌ All requests failed. Check Docker logs for error messages.")

if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    concurrent = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    asyncio.run(run_stress_test(total, concurrent))