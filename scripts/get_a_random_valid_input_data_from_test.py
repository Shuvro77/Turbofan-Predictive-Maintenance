import pandas as pd
import json
import os
import random

# 1. Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
metadata_path = os.path.join(BASE_DIR, 'artifacts', 'metadata.json')
data_path = os.path.join(BASE_DIR, 'data', 'test_FD001.txt')
temp_dir = os.path.join(BASE_DIR, 'temporary')

# Ensure the temporary directory exists
os.makedirs(temp_dir, exist_ok=True)

def generate_random_test_case():
    # 2. Load Metadata (Source of Truth for sensor order)
    try:
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
    except FileNotFoundError:
        print(f"Error: metadata.json not found at {metadata_path}")
        return

    # 3. Load the raw NASA test data
    # NASA FD001 raw format: [0:unit, 1:cycle, 2:set1, 3:set2, 4:set3, 5:s1, 6:s2, ... 25:s21]
    test_df = pd.read_csv(data_path, sep=r'\s+', header=None)

    # 4. Map raw indices to actual sensor names
    raw_column_names = {
        0: 'unit_id', 1: 'cycle', 2: 'set1', 3: 'set2', 4: 'set3',
        5: 's_1', 6: 's_2', 7: 's_3', 8: 's_4', 9: 's_5', 10: 's_6',
        11: 's_7', 12: 's_8', 13: 's_9', 14: 's_10', 15: 's_11',
        16: 's_12', 17: 's_13', 18: 's_14', 19: 's_15', 20: 's_16',
        21: 's_17', 22: 's_18', 23: 's_19', 24: 's_20', 25: 's_21'
    }
    test_df.rename(columns=raw_column_names, inplace=True)

    # 5. Filter for units that have at least enough cycles (50)
    seq_length = meta['sequence_length']
    units = test_df['unit_id'].unique()
    valid_units = [u for u in units if len(test_df[test_df['unit_id'] == u]) >= seq_length]

    if not valid_units:
        print("No units found with enough data cycles.")
        return

    selected_unit = random.choice(valid_units)
    unit_data = test_df[test_df['unit_id'] == selected_unit]

    # 6. Pick a random 50-cycle window from this unit
    max_start = len(unit_data) - seq_length
    start_idx = random.randint(0, max_start)
    window_df = unit_data.iloc[start_idx : start_idx + seq_length]

    # 7. Extract ONLY the sensors defined in metadata.json (in the correct order)
    # This prevents the "Feature names must match" error in the API
    final_data = window_df[meta['sensor_names']]

    # 8. Save to temporary directory
    payload = {"data_window": final_data.to_dict(orient='records')}
    filename = f"random_data_unit_{selected_unit}.json"
    file_path = os.path.join(temp_dir, filename)

    with open(file_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"Success! Random test data for Unit {selected_unit} saved in \"temporary/{filename}\"")
    print(f"This payload contains {seq_length} cycles and {len(meta['sensor_names'])} sensors.")

if __name__ == "__main__":
    generate_random_test_case()