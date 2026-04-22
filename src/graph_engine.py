import json
import pandas as pd
import numpy as np
import os

# Setup
input_path = 'data/raw/nyc_master_2025.csv'
output_dir = 'data/processed/'
os.makedirs(output_dir, exist_ok=True)

GATE_TO_IDX = {
    "EWR_A_Wide":   0,
    "EWR_A_Narrow": 1,
    "EWR_B_Narrow": 2,
    "EWR_C_Wide":   3,
    "EWR_C_Narrow": 4,
}
NUM_GATES = 5

print("Loading master node list...")
df = pd.read_csv(input_path)
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df = df.sort_values(['TAIL_NUM', 'FL_DATE', 'CRS_DEP_TIME'])


# ---------------------------------------------------------------------------
# 1. Turnaround Edges — consecutive flights of the same aircraft
# ---------------------------------------------------------------------------
print("Generating Turnaround Edges (Same Aircraft)...")
edges = []

for tail, group in df.groupby('TAIL_NUM'):
    for i in range(len(group) - 1):
        edges.append({
            'source': group.index[i],
            'target': group.index[i + 1],
            'type': 'turnaround',
        })

# ---------------------------------------------------------------------------
# 2. Congestion Edges — same airport, departure within 15 min
# ---------------------------------------------------------------------------
print("Generating Congestion Edges (Shared Window at EWR)...")

for airport in ['EWR']:
    airport_df = df[df['ORIGIN'] == airport].sort_values('CRS_DEP_TIME')
    for i in range(len(airport_df) - 1):
        for j in range(1, min(10, len(airport_df) - i)):
            t1 = airport_df.iloc[i]['CRS_DEP_TIME']
            t2 = airport_df.iloc[i + j]['CRS_DEP_TIME']
            if abs(t1 - t2) <= 15:
                edges.append({
                    'source': airport_df.index[i],
                    'target': airport_df.index[i + j],
                    'type': 'congestion',
                })

# ---------------------------------------------------------------------------
# 3. Same-Terminal Edges — same airport, within 30 min, shared authorized terminal
#
# Two flights get a same_terminal edge when their carriers are authorized to
# use at least one common terminal.  This gives the GNN a capacity-awareness
# signal: nodes sharing an edge here compete for the same gate space.
# ---------------------------------------------------------------------------
print("Generating Same-Terminal Edges (Shared Terminal Capacity)...")

gate_map = json.load(open('data/meta/gate_mapping.json'))


def _hhmm_to_min(t: int) -> int:
    """Convert HHMM integer (e.g. 1445) to minutes since midnight (e.g. 885)."""
    return (t // 100) * 60 + (t % 100)


_TERMINAL_TO_GATE_INDICES = {
    "Terminal_A": {0, 1},  # EWR_A_Wide + EWR_A_Narrow
    "Terminal_B": {2},     # EWR_B_Narrow
    "Terminal_C": {3, 4},  # EWR_C_Wide + EWR_C_Narrow
}


def _terminal_group(carrier: str, airport: str) -> set:
    """Set of gate indices the carrier is authorized to use at this airport."""
    gates = set()
    for term_name, carriers in gate_map.get(airport, {}).items():
        if carrier in carriers:
            gates |= _TERMINAL_TO_GATE_INDICES.get(term_name, set())
    # Unmapped carrier → all gates are notionally valid (no restriction)
    return gates if gates else set(range(NUM_GATES))


for airport in ['EWR']:
    # Sort by (date, time) so the break condition is correct within each day
    ap_df = (df[df['ORIGIN'] == airport]
             .sort_values(['FL_DATE', 'CRS_DEP_TIME'])
             .reset_index())   # stores original df row index in 'index' column
    groups = [
        _terminal_group(row['OP_UNIQUE_CARRIER'], airport)
        for _, row in ap_df.iterrows()
    ]
    n = len(ap_df)
    for i in range(n):
        t_i    = _hhmm_to_min(ap_df.iloc[i]['CRS_DEP_TIME'])
        date_i = ap_df.iloc[i]['FL_DATE']
        for j in range(i + 1, n):
            if ap_df.iloc[j]['FL_DATE'] != date_i:
                break  # moved to next day — no same-day flights remain
            t_j = _hhmm_to_min(ap_df.iloc[j]['CRS_DEP_TIME'])
            if t_j - t_i > 30:
                break  # sorted ascending — no further j within 30-min window
            if groups[i] & groups[j]:  # carriers share ≥1 authorized terminal
                src = ap_df.iloc[i]['index']
                tgt = ap_df.iloc[j]['index']
                edges.append({'source': src, 'target': tgt, 'type': 'same_terminal'})
                edges.append({'source': tgt, 'target': src, 'type': 'same_terminal'})

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
edges_df = pd.DataFrame(edges)
edges_df.to_csv(os.path.join(output_dir, 'edges.csv'), index=False)

counts = edges_df['type'].value_counts()
print(f"\nGraph Engine Finished!")
print(f"  Turnaround edges  : {counts.get('turnaround', 0):,}")
print(f"  Congestion edges  : {counts.get('congestion', 0):,}")
print(f"  Same-terminal edges: {counts.get('same_terminal', 0):,}")
print(f"  Total             : {len(edges_df):,}")
print(f"Files saved in: {output_dir}")
