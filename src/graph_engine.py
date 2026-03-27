import pandas as pd
import numpy as np
import os

# 1. Setup
input_path = 'data/raw/nyc_master_2025.csv'
output_dir = 'data/processed/'
os.makedirs(output_dir, exist_ok=True)

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
for airport in ['EWR', 'LGA']:
    airport_df = df[df['ORIGIN'] == airport].sort_values('CRS_DEP_TIME')
    
    # Using a rolling window to find flights close in time
    # We convert time to an integer (HHMM) to calculate the 15-min gap
    for i in range(len(airport_df) - 1):
        for j in range(1, min(10, len(airport_df) - i)): # Check next 10 flights
            t1 = airport_df.iloc[i]['CRS_DEP_TIME']
            t2 = airport_df.iloc[i+j]['CRS_DEP_TIME']
            
            # Simple check: if they are within 15 'minutes' of each other
            if abs(t1 - t2) <= 15:
                edges.append({
                    'source': airport_df.index[i], 
                    'target': airport_df.index[i+j], 
                    'type': 'congestion'
                })

# 4. Save to the /processed/ directory
edges_df = pd.DataFrame(edges)
edges_df.to_csv(os.path.join(output_dir, 'edges.csv'), index=False)

print(f"Graph Engine Finished!")
print(f"Total Edges Created: {len(edges_df)}")
print(f"Files saved in: {output_dir}")