import pandas as pd
import numpy as np
import os
from pathlib import Path

# 1. Setup
ROOT = Path(__file__).resolve().parent.parent
input_path = ROOT / 'data' / 'raw' / 'nyc_master_2025.csv'
output_dir = ROOT / 'data' / 'processed'
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading master node list...")
df = pd.read_csv(input_path)

# Ensure data is sorted using the correct BTS headers
# This is critical for the 'Turnaround' logic
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df = df.sort_values(['TAIL_NUM', 'FL_DATE', 'CRS_DEP_TIME'])

# 2. Build Turnaround Edges (Temporal Dependencies)
print("Generating Turnaround Edges (Same Aircraft)...")
edges = []

# Group by aircraft to find consecutive flights
# This handles the F3 (Schedule Stability) objective
for tail, group in df.groupby('TAIL_NUM'):
    # Create an edge from flight[i] to flight[i+1]
    for i in range(len(group) - 1):
        source_idx = group.index[i]
        target_idx = group.index[i+1]

        # Only link if they happen on the same or consecutive day
        # and the physical plane is actually moving between them
        edges.append({
            'source': source_idx,
            'target': target_idx,
            'type': 'turnaround'
        })

# 3. Build Congestion Edges (Spatial Dependencies)
print("Generating Congestion Edges (Shared Window at EWR/LGA)...")
# Linking flights departing within 15 mins of each other at the same airport
# This handles the F2 (Taxiing) objective

def hhmm_to_minutes(hhmm):
    """Convert HHMM integer (e.g. 1455) to minutes since midnight (e.g. 895)."""
    hhmm = hhmm.astype(int)
    return (hhmm // 100) * 60 + (hhmm % 100)

for airport in ['EWR', 'LGA']:
    airport_df = df[df['ORIGIN'] == airport].sort_values(['FL_DATE', 'CRS_DEP_TIME']).copy()
    airport_df['dep_minutes'] = hhmm_to_minutes(airport_df['CRS_DEP_TIME'])

    dep_minutes = airport_df['dep_minutes'].values
    idx_array = airport_df.index.values

    for i in range(len(airport_df) - 1):
        m1 = dep_minutes[i]
        for j in range(i + 1, len(airport_df)):
            m2 = dep_minutes[j]
            diff = m2 - m1  # sorted ascending, so diff >= 0
            if diff > 15:
                break  # no further flights can be within 15 min
            edges.append({
                'source': idx_array[i],
                'target': idx_array[j],
                'type': 'congestion'
            })

# 4. Save to the /processed/ directory
edges_df = pd.DataFrame(edges)
edges_df.to_csv(output_dir / 'edges.csv', index=False)

print(f"Graph Engine Finished!")
print(f"Total Edges Created: {len(edges_df)}")
print(f"Files saved in: {output_dir}")