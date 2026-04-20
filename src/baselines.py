"""
Extended baseline policies for the airport gate scheduler.

Three deterministic greedy baselines:

  GreedyFirstValid  — first authorised terminal (alphabetical).
                      Alias for GreedyFirstFit from baseline.py.

  GreedyMinTaxi     — authorised terminal with shortest taxi distance.
                      Minimises F2 greedily; ignores load.

  GreedyLeastLoaded — authorised terminal with fewest flights in a
                      ±30-min departure window at runtime.
                      Balances terminal utilisation.

All three share the feasibility lookup from GreedyFirstFit and expose
the same assign() / score() interface.

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
    GATE_CLASSES, GATE_TO_IDX, IDX_TO_GATE,
    APPROX_DIST_M, FALLBACK_GATE, AIRPORT_GATE_ORDER,
    NUM_GATES, UNASSIGNED,
)

RAW_CSV  = ROOT / "data" / "raw"  / "nyc_master_2025.csv"
GATE_MAP = ROOT / "data" / "meta" / "gate_mapping.json"


# ---------------------------------------------------------------------------
# Greedy Min-Taxi baseline
# ---------------------------------------------------------------------------

class GreedyMinTaxi(GreedyFirstFit):
    """
    Assign each flight to the authorised terminal with the shortest taxi
    distance to the active runway.

    Among ties in distance, the first gate in airport priority order wins.
    """

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        sort_idx   = np.lexsort((df["CRS_DEP_TIME"].values, df["FL_DATE"].values))
        sorted_df  = df.iloc[sort_idx].reset_index(drop=True)
        assignments = np.full(len(sorted_df), UNASSIGNED, dtype=np.int8)

        for i, row in sorted_df.iterrows():
            airport = self._resolve_airport(row)
            if airport is None:
                continue
            carrier = row["OP_UNIQUE_CARRIER"]
            key     = (carrier, airport)

            candidates = self.valid_gates.get(key)
            if candidates:
                # Pick the candidate with minimum taxi distance
                assignments[i] = min(
                    candidates,
                    key=lambda g: APPROX_DIST_M.get(IDX_TO_GATE.get(g, ""), float("inf")),
                )
            else:
                assignments[i] = FALLBACK_GATE[airport]

        inv_sort = np.argsort(sort_idx)
        return assignments[inv_sort]


# ---------------------------------------------------------------------------
# Greedy Least-Loaded baseline
# ---------------------------------------------------------------------------

class GreedyLeastLoaded(GreedyFirstFit):
    """
    Assign each flight to the authorised terminal that has the fewest
    already-assigned flights in a ±30-minute departure window.

    Flights are processed in chronological order.  A running dict
    window_load[(terminal_idx, time_bucket)] counts assignments per
    30-min slot.  The slot key is the integer (CRS_DEP_TIME // 30).
    """

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        sort_idx    = np.lexsort((df["CRS_DEP_TIME"].values, df["FL_DATE"].values))
        sorted_df   = df.iloc[sort_idx].reset_index(drop=True)
        assignments = np.full(len(sorted_df), UNASSIGNED, dtype=np.int8)

        # load[(gate_idx, bucket)] = count of flights assigned so far
        load: dict = defaultdict(int)

        for i, row in sorted_df.iterrows():
            airport = self._resolve_airport(row)
            if airport is None:
                continue
            carrier = row["OP_UNIQUE_CARRIER"]
            key     = (carrier, airport)
            bucket  = int(row["CRS_DEP_TIME"]) // 30

            candidates = self.valid_gates.get(key)
            if candidates:
                chosen = min(candidates, key=lambda g: load[(g, bucket)])
            else:
                chosen = FALLBACK_GATE[airport]

            assignments[i] = chosen
            load[(chosen, bucket)] += 1

        inv_sort = np.argsort(sort_idx)
        return assignments[inv_sort]


# ---------------------------------------------------------------------------
# Greedy Balanced baseline
# ---------------------------------------------------------------------------

class GreedyBalanced(GreedyFirstFit):
    """
    Assign each flight to the authorised terminal minimising:

        score(g) = alpha * norm_dist(g) + (1 - alpha) * norm_load(g, bucket)

    where norm_dist is the terminal's taxi distance normalised to [0,1]
    within the airport, and norm_load = load / (1 + load) for the 30-min
    departure window.

    alpha=0.5 gives equal weight to taxi efficiency and load balance.
    This represents a manually-tuned balanced heuristic; the GNN learns
    the same trade-off automatically from congestion and same_terminal
    edges without any hyperparameter tuning.
    """

    def __init__(self, gate_mapping_path: str = str(GATE_MAP), alpha: float = 0.5):
        super().__init__(gate_mapping_path)
        self.alpha = alpha
        max_ewr = max(APPROX_DIST_M[k] for k in ["EWR_Terminal_A", "EWR_Terminal_B", "EWR_Terminal_C"])
        max_lga = max(APPROX_DIST_M[k] for k in ["LGA_Terminal_B", "LGA_Terminal_C"])
        self._norm_dist = {}
        for g, name in IDX_TO_GATE.items():
            mx = max_ewr if "EWR" in name else max_lga
            self._norm_dist[g] = APPROX_DIST_M[name] / mx

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        sort_idx  = np.lexsort((df["CRS_DEP_TIME"].values, df["FL_DATE"].values))
        sorted_df = df.iloc[sort_idx].reset_index(drop=True)
        assignments = np.full(len(sorted_df), UNASSIGNED, dtype=np.int8)
        load: dict = defaultdict(int)

        for i, row in sorted_df.iterrows():
            airport = self._resolve_airport(row)
            if airport is None:
                continue
            carrier = row["OP_UNIQUE_CARRIER"]
            key     = (carrier, airport)
            bucket  = int(row["CRS_DEP_TIME"]) // 30

            candidates = self.valid_gates.get(key)
            if candidates:
                def _score(g, _load=load, _bucket=bucket):
                    nd = self._norm_dist[g]
                    ld = _load[(g, _bucket)] / (1 + _load[(g, _bucket)])
                    return self.alpha * nd + (1 - self.alpha) * ld
                chosen = min(candidates, key=_score)
            else:
                chosen = FALLBACK_GATE[airport]

            assignments[i] = chosen
            load[(chosen, bucket)] += 1

        return assignments[np.argsort(sort_idx)]


# ---------------------------------------------------------------------------
# Unified comparison
# ---------------------------------------------------------------------------

def _delay_risk(df: pd.DataFrame) -> float:
    """Fraction of flights with DEP_DELAY > 15 min."""
    delays = pd.to_numeric(df.get("DEP_DELAY", pd.Series()), errors="coerce").fillna(0)
    return float((delays > 15).mean())


def _utilisation_balance(assignments: np.ndarray) -> float:
    """
    Coefficient of variation of terminal counts (std / mean).
    Lower = more balanced utilisation across terminals.
    """
    counts = np.array(
        [(assignments == g).sum() for g in range(NUM_GATES)],
        dtype=float,
    )
    mean = counts.mean()
    if mean == 0:
        return 0.0
    return float(counts.std() / mean)


def _pct_different(a: np.ndarray, b: np.ndarray) -> float:
    """Percentage of flights where assignments a and b disagree."""
    both_valid = (a != UNASSIGNED) & (b != UNASSIGNED)
    if both_valid.sum() == 0:
        return 0.0
    return float((a[both_valid] != b[both_valid]).mean() * 100)


def compare_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all three baselines on df and return a comparison DataFrame.

    Columns: policy, f2_mean_dist_m, f3_delay_risk_pct,
             utilisation_cv, pct_changed_vs_first_valid
    """
    policies = {
        "GreedyFirstValid  (pi0)": GreedyFirstFit(str(GATE_MAP)),
        "GreedyMinTaxi         ": GreedyMinTaxi(str(GATE_MAP)),
        "GreedyLeastLoaded     ": GreedyLeastLoaded(str(GATE_MAP)),
        "GreedyBalanced (a=0.5)": GreedyBalanced(str(GATE_MAP)),
    }

    results   = {}
    reference = None

    for name, policy in policies.items():
        assignments = policy.assign(df)
        metrics     = policy.score(assignments, df)
        results[name] = {
            "assignments": assignments,
            "f2"         : metrics["f2_mean_dist_m"],
            "f3_risk"    : _delay_risk(df) * 100,
            "util_cv"    : _utilisation_balance(assignments),
        }
        if reference is None:
            reference = assignments

    rows = []
    for name, r in results.items():
        rows.append({
            "policy"                      : name.strip(),
            "f2_mean_dist_m"              : round(r["f2"], 1),
            "f3_delay_risk_pct"           : round(r["f3_risk"], 2),
            "utilisation_cv"              : round(r["util_cv"], 4),
            "pct_changed_vs_first_valid"  : round(
                _pct_different(r["assignments"], reference), 2
            ),
        })

    return pd.DataFrame(rows)


def print_comparison(cmp_df: pd.DataFrame) -> None:
    sep = "-" * 80
    print(f"\n{'=' * 80}")
    print("  Baseline Policy Comparison")
    print(f"{'=' * 80}")
    print(f"  {'Policy':<30}  {'F2 dist(m)':>10}  {'F3 risk%':>9}  "
          f"{'Util CV':>8}  {'%d vs pi0':>9}")
    print(sep)
    for _, row in cmp_df.iterrows():
        print(f"  {row['policy']:<30}  {row['f2_mean_dist_m']:>10.1f}  "
              f"{row['f3_delay_risk_pct']:>9.2f}  "
              f"{row['utilisation_cv']:>8.4f}  "
              f"{row['pct_changed_vs_first_valid']:>9.2f}")
    print(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading flight data…")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    print(f"  {len(df):,} flights loaded.\n")

    cmp = compare_baselines(df)
    print_comparison(cmp)

    out = ROOT / "outputs" / "baseline_comparison.csv"
    out.parent.mkdir(exist_ok=True)
    cmp.to_csv(out, index=False)
    print(f"Saved -> {out}")
