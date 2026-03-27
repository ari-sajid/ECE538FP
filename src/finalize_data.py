import pandas as pd
import numpy as np

# 1. Load Data
print("Loading flight and weather data...")
flights = pd.read_csv('data/raw/nyc_master_2025.csv')
weather = pd.read_csv('data/processed/weather_clean_2025.csv')

# 2. Prepare Timestamps for Merging
# We round flight times to the nearest hour to match the weather observations
flights['FL_DATE'] = pd.to_datetime(flights['FL_DATE'])
flights['hour_group'] = pd.to_datetime(
    flights['FL_DATE'].dt.strftime('%Y-%m-%d') + ' ' + 
    (flights['CRS_DEP_TIME'] // 100).astype(str).str.zfill(2) + ':00:00'
)
weather['valid'] = pd.to_datetime(weather['valid'])

# 3. Merge Flight Nodes with Weather
print("Merging weather features into flight nodes...")
# We match weather at the ORIGIN airport
df = pd.merge(
    flights, 
    weather, 
    left_on=['hour_group', 'ORIGIN'], 
    right_on=['valid', 'station'], 
    how='left'
)

# 4. Encode Categorical Features
print("Encoding airlines and airports...")
# Convert Airline and Origin into numeric codes for the GNN
df = pd.get_dummies(df, columns=['OP_UNIQUE_CARRIER', 'ORIGIN'])

# 5. Final Cleanup
# Drop intermediate columns and save the "Gold" dataset
cols_to_drop = ['hour_group', 'valid', 'station', 'DEST', 'TAIL_NUM']
final_df = df.drop(columns=cols_to_drop)

final_df.to_csv('data/processed/final_node_features.csv', index=False)
print(f"Success! Final Node Matrix saved with {final_df.shape[1]} features.")