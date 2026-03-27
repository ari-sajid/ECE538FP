import pandas as pd
import zipfile
import os

# 1. Paths and Configuration
zip_path = 'data/raw/2025_BTS.zip'
output_path = 'data/raw/nyc_master_2025.csv'
target_airports = ['EWR', 'LGA']

# 2. UPDATED: Using the exact BTS internal headers
cols_to_load = [
    'FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN', 'DEST',
    'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'TAXI_IN',
    'WHEELS_OFF', 'WHEELS_ON', 'DISTANCE', 'LATE_AIRCRAFT_DELAY'
]

all_filtered_data = []

# 3. Process the Zip
if not os.path.exists(zip_path):
    print(f"Error: Could not find {zip_path}")
else:
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Looking into the nested folder structure you specified
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        print(f"Found {len(csv_files)} files to process.")

        for csv_file in csv_files:
            print(f"Processing: {csv_file}")
            with z.open(csv_file) as f:
                # Chunking to keep RAM usage low
                for chunk in pd.read_csv(f, usecols=cols_to_load, chunksize=100000, low_memory=False):
                    # Filtering using the new all-caps headers
                    filtered_chunk = chunk[
                        (chunk['ORIGIN'].isin(target_airports)) | 
                        (chunk['DEST'].isin(target_airports))
                    ].copy()
                    
                    all_filtered_data.append(filtered_chunk)

# 4. Combine, Sort, and Export
if all_filtered_data:
    print("Unifying all months into master CSV...")
    master_df = pd.concat(all_filtered_data, ignore_index=True)

    # Sort update
    master_df['FL_DATE'] = pd.to_datetime(master_df['FL_DATE'])
    master_df = master_df.sort_values(by=['FL_DATE', 'CRS_DEP_TIME'])

    # Fill NaNs for the delay cause column
    master_df['LATE_AIRCRAFT_DELAY'] = master_df['LATE_AIRCRAFT_DELAY'].fillna(0)

    master_df.to_csv(output_path, index=False)
    print(f"Success! Master file created at {output_path}")
    print(f"Total Flight Nodes: {len(master_df)}")
else:
    print("No data matched your criteria. Check your airport codes.")