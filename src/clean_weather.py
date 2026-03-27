import pandas as pd
import numpy as np
import os

def process_weather(file_path, station_name):
    print(f"Cleaning weather for {station_name}...")
    df = pd.read_csv(file_path)
    
    # 1. Clean 'M' (Missing) and 'T' (Trace) values
    df = df.replace('M', np.nan)
    df['p01i'] = df['p01i'].replace('T', 0.005)
    
    # 2. Convert specific columns to numeric
    cols_to_fix = ['tmpf', 'drct', 'sped', 'alti', 'p01i', 'vsby']
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. Handle Time
    df['valid'] = pd.to_datetime(df['valid'])
    # Round to the nearest hour using 'h' to avoid the FutureWarning
    df['valid'] = df['valid'].dt.round('h')
    
    # 4. Group by hour and aggregate
    # Specify mean for numeric columns and 'first' for the station identifier
    agg_dict = {col: 'mean' for col in cols_to_fix}
    agg_dict['station'] = 'first'
    
    df = df.groupby('valid').agg(agg_dict).reset_index()
    
    # 5. Impute missing values (Forward Fill then Backward Fill)
    df = df.sort_values('valid').ffill().bfill()
    
    # 6. Feature Scaling (Min-Max)
    # This ensures features are on the same scale for GNN stability
    for col in ['sped', 'vsby', 'p01i']:
        max_val = df[col].max()
        min_val = df[col].min()
        if max_val != min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0.0
            
    return df

# Ensure output directory exists
os.makedirs('data/processed/', exist_ok=True)

# Process both airports and combine into a single file
ewr_clean = process_weather('data/meta/weather_ewr.csv', 'EWR')
lga_clean = process_weather('data/meta/weather_lga.csv', 'LGA')

weather_master = pd.concat([ewr_clean, lga_clean])
weather_master.to_csv('data/processed/weather_clean_2025.csv', index=False)
print("Success! Processed weather saved to data/processed/weather_clean_2025.csv")