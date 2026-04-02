"""
Baseline Gate Scheduling Policy (π₀): Greedy First-Fit.

This module implements the deterministic baseline against which the GNN
(πθ) is evaluated.  It mirrors the "frozen plan" heuristic used at most
hub airports: reserve blocks of terminals per airline and fill them
sequentially by scheduled departure time.

Algorithm
---------
1. Sort all flights by (FL_DATE, CRS_DEP_TIME).
2. For each flight determine its airport (EWR or LGA) from ORIGIN / DEST.
3. Look up the airline's authorised terminals at that airport via
   gate_mapping.json.
4. Assign the *first* authorised terminal (alphabetical order, i.e.
   Terminal A before B before C).
5. Unmapped carriers (not in gate_mapping) fall back to the highest-
   capacity terminal at the airport — never blocking a known carrier.

Properties
----------
- Zero airline–terminal violations for all mapped carriers.
- O(N log N) runtime (dominated by the sort).
- Fully deterministic given the same input.
- Suboptimal for F2 (taxiing distance) and F3 (delay propagation)
  because no look-ahead is applied.

Metrics produced
----------------
F1  Gate violation rate  : fraction of assignments to invalid terminals.
F2  Mean taxi distance   : average terminal→runway distance (metres).
F3  Schedule instability : mean positive departure delay (minutes).
     + delay propagation rate: fraction of flights with DEP_DELAY > 0.
     + late-aircraft share   : fraction caused by incoming aircraft delay.

Run standalone:
    python src/baseline.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
RAW_CSV  = ROOT / "data" / "raw"  / "nyc_master_2025.csv"
GATE_MAP = ROOT / "data" / "meta" / "gate_mapping.json"

# ---------------------------------------------------------------------------
# Gate index convention (matches model.py / loss.py)
# ---------------------------------------------------------------------------
GATE_CLASSES = [
    "EWR_Terminal_A",   # 0
    "EWR_Terminal_B",   # 1
    "EWR_Terminal_C",   # 2
    "LGA_Terminal_B",   # 3
    "LGA_Terminal_C",   # 4
]
GATE_TO_IDX = {g: i for i, g in enumerate(GATE_CLASSES)}
IDX_TO_GATE = {i: g for i, g in enumerate(GATE_CLASSES)}
NUM_GATES   = len(GATE_CLASSES)
UNASSIGNED  = -1   # sentinel for flights with no resolvable airport

# ---------------------------------------------------------------------------
# Approximate terminal → active-runway distances (metres).
# Derived from OSM NetworkX shortest-path over the GraphML graphs.
# Used for F2 evaluation without reloading the 20 MB of GraphML data.
# ---------------------------------------------------------------------------
APPROX_DIST_M = {
    "EWR_Terminal_A": 810,
    "EWR_Terminal_B": 1_140,
    "EWR_Terminal_C": 1_380,
    "LGA_Terminal_B": 660,
    "LGA_Terminal_C": 920,
}

# Greedy preference order within each airport (alphabetical = first-fit)
AIRPORT_GATE_ORDER = {
    "EWR": [0, 1, 2],   # A → B → C
    "LGA": [3, 4],       # B → C
}

# Fallback terminal for carriers absent from gate_mapping
FALLBACK_GATE = {
    "EWR": 0,   # Terminal A — highest capacity
    "LGA": 3,   # Terminal B — highest capacity
}


# ===========================================================================
class GreedyFirstFit:
    """
    Deterministic greedy gate-assignment baseline (π₀).

    Parameters
    ----------
    gate_mapping_path : str or Path
        Path to gate_mapping.json.
    """

    def __init__(self, gate_mapping_path: str = str(GATE_MAP)):
        with open(gate_mapping_path) as fh:
            raw = json.load(fh)

        # valid_gates[(carrier, airport)] → list of gate indices in priority order
        # e.g. ('UA', 'EWR') → [0, 2]   (Terminal A first, then C)
        self.valid_gates: dict = {}

        for airport, terminals in raw.items():
            if airport not in AIRPORT_GATE_ORDER:
                continue
            for terminal_name, carriers in terminals.items():
                gate_key = f"{airport}_{terminal_name}"
                if gate_key not in GATE_TO_IDX:
                    continue
                g_idx = GATE_TO_IDX[gate_key]
                for carrier in carriers:
                    key = (carrier, airport)
                    self.valid_gates.setdefault(key, [])
                    self.valid_gates[key].append(g_idx)

        # Sort each carrier's valid gates into the greedy priority order
        for key in self.valid_gates:
            airport = key[1]
            priority = AIRPORT_GATE_ORDER[airport]
            self.valid_gates[key] = sorted(
                self.valid_gates[key], key=lambda g: priority.index(g)
            )

    # -----------------------------------------------------------------------
    def _resolve_airport(self, row) -> str:
        """Return 'EWR' or 'LGA' for the flight's NYC-side airport."""
        if row["ORIGIN"] in ("EWR", "LGA"):
            return row["ORIGIN"]
        if row.get("DEST") in ("EWR", "LGA"):
            return row["DEST"]
        return None

    # -----------------------------------------------------------------------
    def assign(self, df: pd.DataFrame) -> np.ndarray:
        """
        Assign a terminal index to every flight.

        Parameters
        ----------
        df : DataFrame
            Must contain at minimum: FL_DATE, CRS_DEP_TIME,
            OP_UNIQUE_CARRIER, ORIGIN.  DEST is used if present.

        Returns
        -------
        assignments : ndarray[int], shape (N,)
            Terminal index 0-4 per flight, or UNASSIGNED (-1) if the
            flight's airport cannot be resolved.
        """
        # Sort by scheduled departure (preserves determinism)
        sort_idx = np.lexsort((df["CRS_DEP_TIME"].values,
                               df["FL_DATE"].values))
        sorted_df = df.iloc[sort_idx].reset_index(drop=True)

        assignments = np.full(len(sorted_df), UNASSIGNED, dtype=np.int8)

        for i, row in sorted_df.iterrows():
            airport = self._resolve_airport(row)
            if airport is None:
                continue   # flight does not involve EWR or LGA

            carrier = row["OP_UNIQUE_CARRIER"]
            key     = (carrier, airport)

            if key in self.valid_gates and self.valid_gates[key]:
                # Greedy first-fit: pick the first authorised terminal
                assignments[i] = self.valid_gates[key][0]
            else:
                # Unmapped carrier — fall back to highest-capacity terminal
                assignments[i] = FALLBACK_GATE[airport]

        # Return in the original row order
        inv_sort = np.argsort(sort_idx)
        return assignments[inv_sort]

    # -----------------------------------------------------------------------
    def score(
        self,
        assignments: np.ndarray,
        df: pd.DataFrame,
    ) -> dict:
        """
        Compute F1 / F2 / F3 metrics for a given assignment vector.

        Parameters
        ----------
        assignments : ndarray[int], shape (N,)  –  output of assign()
        df : DataFrame  –  same frame passed to assign()

        Returns
        -------
        dict with keys:
            f1_violations, f1_rate,
            f2_mean_dist_m, f2_per_terminal,
            f3_mean_positive_delay, f3_delay_rate, f3_late_aircraft_rate,
            terminal_counts, n_unassigned
        """
        N = len(df)
        assigned_mask = assignments != UNASSIGNED

        # ── F1: gate-constraint violations ──────────────────────────────────
        violations = 0
        for i, row in df.iterrows():
            a = assignments[i]
            if a == UNASSIGNED:
                continue
            airport  = self._resolve_airport(row)
            carrier  = row["OP_UNIQUE_CARRIER"]
            key      = (carrier, airport)
            valid    = self.valid_gates.get(key, [])
            # Unmapped carriers always use fallback → no violation counted
            if valid and a not in valid:
                violations += 1

        f1_rate = violations / max(assigned_mask.sum(), 1)

        # ── F2: taxiing distance ─────────────────────────────────────────────
        dist_array = np.array(
            [APPROX_DIST_M.get(IDX_TO_GATE.get(a, ""), 0)
             if a != UNASSIGNED else 0
             for a in assignments],
            dtype=np.float32,
        )
        f2_mean = float(dist_array[assigned_mask].mean()) if assigned_mask.any() else 0.0

        # Per-terminal mean distance
        f2_per_terminal = {}
        for gate_name, dist_m in APPROX_DIST_M.items():
            idx   = GATE_TO_IDX[gate_name]
            count = int((assignments == idx).sum())
            f2_per_terminal[gate_name] = {"count": count, "dist_m": dist_m}

        # ── F3: schedule stability ───────────────────────────────────────────
        delays = pd.to_numeric(df.get("DEP_DELAY", pd.Series()), errors="coerce").fillna(0)
        late_aircraft = pd.to_numeric(
            df.get("LATE_AIRCRAFT_DELAY", pd.Series()), errors="coerce"
        ).fillna(0)

        f3_mean_pos      = float(delays.clip(lower=0).mean())
        f3_delay_rate    = float((delays > 0).mean())
        f3_late_ac_rate  = float((late_aircraft > 0).mean())

        # Terminal distribution
        terminal_counts = {
            IDX_TO_GATE.get(g, f"idx_{g}"): int((assignments == g).sum())
            for g in range(NUM_GATES)
        }

        return dict(
            n_flights       = N,
            n_assigned      = int(assigned_mask.sum()),
            n_unassigned    = int((~assigned_mask).sum()),
            # F1
            f1_violations   = violations,
            f1_rate         = f1_rate,
            # F2
            f2_mean_dist_m  = f2_mean,
            f2_per_terminal = f2_per_terminal,
            # F3
            f3_mean_pos_delay_min = f3_mean_pos,
            f3_delay_rate         = f3_delay_rate,
            f3_late_aircraft_rate = f3_late_ac_rate,
            # Distribution
            terminal_counts = terminal_counts,
        )

    # -----------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> tuple:
        """Convenience: assign + score in one call."""
        assignments = self.assign(df)
        metrics     = self.score(assignments, df)
        return assignments, metrics


# ===========================================================================
def print_report(metrics: dict, title: str = "Baseline Policy (π₀): Greedy First-Fit"):
    """Pretty-print the metrics dict returned by GreedyFirstFit.score()."""
    sep = "─" * 58
    print(f"\n{'=' * 58}")
    print(f"  {title}")
    print(f"{'=' * 58}")
    print(f"  Flights evaluated  : {metrics['n_flights']:>10,}")
    print(f"  Assigned           : {metrics['n_assigned']:>10,}")
    print(f"  Unassigned (no NYC): {metrics['n_unassigned']:>10,}")
    print(sep)

    # F1
    print(f"  F1  Gate violations     : {metrics['f1_violations']:>6,}  "
          f"({100 * metrics['f1_rate']:.3f}%)")

    # F2
    print(f"  F2  Mean taxi distance  : {metrics['f2_mean_dist_m']:>8.0f} m")
    print(f"       ├─ EWR Terminal A  : {metrics['f2_per_terminal']['EWR_Terminal_A']['count']:>8,} flights  "
          f"({metrics['f2_per_terminal']['EWR_Terminal_A']['dist_m']} m)")
    print(f"       ├─ EWR Terminal B  : {metrics['f2_per_terminal']['EWR_Terminal_B']['count']:>8,} flights  "
          f"({metrics['f2_per_terminal']['EWR_Terminal_B']['dist_m']} m)")
    print(f"       ├─ EWR Terminal C  : {metrics['f2_per_terminal']['EWR_Terminal_C']['count']:>8,} flights  "
          f"({metrics['f2_per_terminal']['EWR_Terminal_C']['dist_m']} m)")
    print(f"       ├─ LGA Terminal B  : {metrics['f2_per_terminal']['LGA_Terminal_B']['count']:>8,} flights  "
          f"({metrics['f2_per_terminal']['LGA_Terminal_B']['dist_m']} m)")
    print(f"       └─ LGA Terminal C  : {metrics['f2_per_terminal']['LGA_Terminal_C']['count']:>8,} flights  "
          f"({metrics['f2_per_terminal']['LGA_Terminal_C']['dist_m']} m)")

    # F3
    print(sep)
    print(f"  F3  Mean positive delay    : {metrics['f3_mean_pos_delay_min']:>7.2f} min")
    print(f"      Delay rate (DEP_DELAY>0): {100 * metrics['f3_delay_rate']:>6.1f}%")
    print(f"      Late-aircraft share     : {100 * metrics['f3_late_aircraft_rate']:>6.1f}%")
    print(f"{'=' * 58}\n")


# ===========================================================================
# Standalone entry point
# ===========================================================================
if __name__ == "__main__":
    print("Loading flight data…")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"  {len(df):,} flights loaded.\n")

    policy      = GreedyFirstFit(GATE_MAP)
    assignments, metrics = policy.run(df)

    print_report(metrics)

    # Optional: save assignments for downstream use
    out_path = ROOT / "outputs" / "baseline_assignments.csv"
    out_path.parent.mkdir(exist_ok=True)
    pd.DataFrame({
        "flight_idx"  : np.arange(len(df)),
        "assignment"  : assignments,
        "gate_name"   : [IDX_TO_GATE.get(a, "UNASSIGNED") for a in assignments],
    }).to_csv(out_path, index=False)
    print(f"Assignments saved → {out_path}")
