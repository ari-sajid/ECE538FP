"""
Operational baseline policy for the EWR gate scheduler.

GreedyCapacityAware — single smart baseline that:
  1. Respects carrier authorization at terminal level
  2. Enforces aircraft-type constraints (widebody → Wide gates only)
  3. Load-balances by slot utilization in 30-min departure windows
  4. Penalizes narrowbody use of Wide gates to preserve wide gate capacity

Run standalone:
    python src/baselines.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import (                          # noqa: E402
    GreedyFirstFit, print_report,
    GATE_CLASSES, GATE_TO_IDX,
    APPROX_DIST_M, FALLBACK_GATE, AIRPORT_GATE_ORDER,
    GATE_SLOTS, IS_NARROW_GATE,
    UNASSIGNED,
)

RAW_CSV  = ROOT / "data" / "raw"  / "nyc_master_2025.csv"
GATE_MAP = ROOT / "data" / "meta" / "gate_mapping.json"

_MAX_DIST_M = max(APPROX_DIST_M.values())  # used for distance normalisation


# ---------------------------------------------------------------------------
# GreedyCapacityAware — single operational baseline
# ---------------------------------------------------------------------------

class GreedyCapacityAware(GreedyFirstFit):
    """
    Single operational baseline that balances taxi efficiency and gate load.

    Scoring for gate class g given flight i:
        score(g) = ALPHA * norm_dist(g)
                 + (1 - ALPHA) * slot_utilization(g, bucket)
                 + WIDE_PENALTY  [if narrowbody and g is Wide]

    - norm_dist = APPROX_DIST_M[g] / max_dist  ∈ [0, 1]
    - slot_utilization = slots_used(g, bucket) / GATE_SLOTS[g]  ∈ [0, ∞)
    - WIDE_PENALTY biases narrowbodies toward Narrow gates to preserve Wide
      gate capacity for widebody aircraft.

    Tracks slot units consumed (widebody = 2, narrowbody = 1) per 30-min bucket.
    """

    WIDE_PENALTY = 0.3   # added to score when narrowbody uses a Wide gate
    ALPHA        = 0.4   # weight of distance vs slot utilization

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        sort_idx  = np.lexsort((df["CRS_DEP_TIME"].values, df["FL_DATE"].values))
        sorted_df = df.iloc[sort_idx].reset_index(drop=True)
        assignments = np.full(len(sorted_df), UNASSIGNED, dtype=np.int8)

        # load_units[(gate_idx, bucket)] = slot-units assigned so far
        load_units: dict = defaultdict(int)

        widebody_col = sorted_df.get("WIDEBODY", pd.Series(0, index=sorted_df.index))

        for i, row in sorted_df.iterrows():
            airport = self._resolve_airport(row)
            if airport is None:
                continue

            carrier   = row["OP_UNIQUE_CARRIER"]
            is_wide   = bool(widebody_col.iloc[i])
            key       = (carrier, airport)
            bucket    = int(row["CRS_DEP_TIME"]) // 30
            candidates = list(self.valid_gates.get(key, []))

            if is_wide:
                # Widebody: only Wide gate classes
                candidates = [g for g in candidates if not IS_NARROW_GATE[GATE_CLASSES[g]]]
                if not candidates:
                    chosen = 0   # EWR_A_Wide fallback for widebody
                    assignments[i] = chosen
                    load_units[(chosen, bucket)] += 2
                    continue

            if not candidates:
                chosen = FALLBACK_GATE[airport]
                assignments[i] = chosen
                load_units[(chosen, bucket)] += 2 if is_wide else 1
                continue

            slots_needed = 2 if is_wide else 1

            def _score(g, _lu=load_units, _b=bucket, _wide=is_wide):
                util       = _lu[(g, _b)] / GATE_SLOTS[GATE_CLASSES[g]]
                dist_score = APPROX_DIST_M[GATE_CLASSES[g]] / _MAX_DIST_M
                penalty    = self.WIDE_PENALTY if (not _wide and not IS_NARROW_GATE[GATE_CLASSES[g]]) else 0.0
                return self.ALPHA * dist_score + (1 - self.ALPHA) * util + penalty

            chosen = min(candidates, key=_score)
            assignments[i] = chosen
            load_units[(chosen, bucket)] += slots_needed

        return assignments[np.argsort(sort_idx)]


# ---------------------------------------------------------------------------
# Standalone comparison
# ---------------------------------------------------------------------------

def run_comparison(df: pd.DataFrame) -> None:
    sep = "─" * 72
    policy      = GreedyCapacityAware(str(GATE_MAP))
    assignments = policy.assign(df)
    metrics     = policy.score(assignments, df)

    print(f"\n{'=' * 72}")
    print("  GreedyCapacityAware — operational baseline")
    print(f"{'=' * 72}")
    print(f"  {'Gate class':<20}  {'Count':>8}  {'Dist(m)':>8}  {'Slots':>6}")
    print(sep)
    tc = metrics["terminal_counts"]
    for g_name in GATE_CLASSES:
        cnt = tc.get(g_name, 0)
        print(f"  {g_name:<20}  {cnt:>8,}  {APPROX_DIST_M[g_name]:>8}  {GATE_SLOTS[g_name]:>6}")
    print(sep)
    print(f"  F1 violations     : {metrics['f1_violations']:,}  ({100*metrics['f1_rate']:.3f}%)")
    print(f"  F2 mean taxi dist : {metrics['f2_mean_dist_m']:.1f} m")
    print(f"  F3 taxi time      : {metrics['f3_taxi_min']:.2f} min")
    print(f"  F3 queue wait     : {metrics['f3_queue_min']:.2f} min")
    print(f"  F3 total delay    : {metrics['f3_total_min']:.2f} min")
    print(f"{'=' * 72}\n")

    out = ROOT / "outputs" / "baseline_comparison.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame([{
        "policy"          : "GreedyCapacityAware",
        "f2_mean_dist_m"  : round(metrics["f2_mean_dist_m"], 1),
        "f3_taxi_min"     : round(metrics["f3_taxi_min"], 3),
        "f3_queue_min"    : round(metrics["f3_queue_min"], 3),
        "f3_total_min"    : round(metrics["f3_total_min"], 3),
        "f1_violations"   : metrics["f1_violations"],
    }]).to_csv(out, index=False)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading flight data…")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    print(f"  {len(df):,} flights loaded.\n")

    run_comparison(df)
