"""
Baseline Gate Scheduling Policy (π₀): Greedy First-Fit.

EWR-only.  Assigns each flight to the first authorised gate class
(priority order: A_Wide → A_Narrow → B_Narrow → C_Wide → C_Narrow)
sorted by scheduled departure time.

Widebody aircraft are restricted to Wide gate classes only.
Narrowbody aircraft may use any authorised gate class.

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
    "EWR_A_Wide",    # 0 — Terminal A widebody gates
    "EWR_A_Narrow",  # 1 — Terminal A narrowbody gates
    "EWR_B_Narrow",  # 2 — Terminal B narrowbody gates
    "EWR_C_Wide",    # 3 — Terminal C widebody gates
    "EWR_C_Narrow",  # 4 — Terminal C narrowbody gates
]
GATE_TO_IDX = {g: i for i, g in enumerate(GATE_CLASSES)}
IDX_TO_GATE = {i: g for i, g in enumerate(GATE_CLASSES)}
NUM_GATES   = len(GATE_CLASSES)
UNASSIGNED  = -1   # sentinel for flights with no resolvable airport

# Approximate terminal → active-runway distances (metres).
APPROX_DIST_M = {
    "EWR_A_Wide":   810,
    "EWR_A_Narrow": 810,
    "EWR_B_Narrow": 1140,
    "EWR_C_Wide":   1380,
    "EWR_C_Narrow": 1380,
}

# Total NB-equivalent slots per gate class (K in M/D/K model)
GATE_SLOTS = {
    "EWR_A_Wide":   8,   # 4 physical gates × 2 slots each
    "EWR_A_Narrow": 29,
    "EWR_B_Narrow": 9,
    "EWR_C_Wide":   4,   # 2 physical gates × 2 slots each
    "EWR_C_Narrow": 55,
}

# True for gate-class indices that are narrow (cannot serve widebody aircraft)
IS_NARROW_GATE = {g: ("Narrow" in g) for g in GATE_CLASSES}

# Terminal name → gate class indices (for building valid_gates)
_TERMINAL_TO_GATE_INDICES = {
    "Terminal_A": [0, 1],  # EWR_A_Wide, EWR_A_Narrow
    "Terminal_B": [2],     # EWR_B_Narrow
    "Terminal_C": [3, 4],  # EWR_C_Wide, EWR_C_Narrow
}

# Greedy preference order within each airport
AIRPORT_GATE_ORDER = {
    "EWR": [0, 1, 2, 3, 4],
}

# Fallback gate class for carriers absent from gate_mapping
FALLBACK_GATE = {
    "EWR": 1,   # EWR_A_Narrow (most total capacity)
}

# ---------------------------------------------------------------------------
# Queueing simulation constants
# ---------------------------------------------------------------------------
TAXI_SPEED_M_PER_MIN   = 200.0   # ~12 km/h conservative ground speed
MIN_TURNAROUND_MIN     = 45.0    # gate occupied this long before next departure
GATE_OCCUPY_WINDOW_MIN = 120.0   # total gate occupancy window per flight

# Density-delay constants: planes parked near each other reduce maneuvering speed.
# Extra delay = DENSITY_FACTOR * max(0, fill_ratio - DENSITY_THRESHOLD)^2
DENSITY_FACTOR    = 8.0    # max extra minutes at full (100%) zone occupancy
DENSITY_THRESHOLD = 0.30   # fill fraction below which no penalty applies
DENSITY_WINDOW_MIN = 60.0  # look-around window for concurrent occupancy count


def simulate_queuing_f3(
    assignments: np.ndarray,      # [N] gate class index (-1 = unassigned)
    dep_times_min: np.ndarray,    # [N] departure time in minutes since epoch
    is_widebody_arr: np.ndarray,  # [N] 1 = widebody, 0 = narrowbody
) -> np.ndarray:
    """
    Slot-based M/D/K queueing simulation with density-proximity penalty.

    Each gate class has GATE_SLOTS[class] independent NB-equivalent slots.
    Widebody aircraft consume 2 consecutive slots; narrowbody consumes 1.
    Flights sorted by dep_time; each requests slots MIN_TURNAROUND_MIN before
    scheduled departure.  If insufficient slots are free: wait.

    Additionally, a density delay is added: when many planes occupy the same
    gate zone concurrently (within DENSITY_WINDOW_MIN), reduced maneuvering
    room slows ground operations.  Penalty is quadratic in fill ratio above
    DENSITY_THRESHOLD.

    Returns
    -------
    wait_times : ndarray[float], shape (N,)  — per-flight gate wait time (min).
    """
    wait_times = np.zeros(len(assignments), dtype=np.float64)
    # Each gate class has a sorted list of per-slot free times
    gate_free: dict = {
        t: sorted([0.0] * GATE_SLOTS[GATE_CLASSES[t]])
        for t in range(NUM_GATES)
    }

    for idx in np.argsort(dep_times_min):
        a = int(assignments[idx])
        if a == UNASSIGNED:
            continue
        slots_needed = 2 if is_widebody_arr[idx] else 1
        t_request    = float(dep_times_min[idx]) - MIN_TURNAROUND_MIN
        free_list    = gate_free[a]

        # Need `slots_needed` earliest-available slots simultaneously.
        # free_list is sorted: free_list[slots_needed - 1] is the (slots_needed)-th
        # earliest free time — both slots must be available by that time.
        t_avail = free_list[slots_needed - 1]
        t_start = max(t_request, t_avail)
        wait_times[idx] = max(0.0, t_start - t_request)

        # Mark consumed slots as busy until t_start + GATE_OCCUPY_WINDOW_MIN
        for s in range(slots_needed):
            free_list[s] = t_start + GATE_OCCUPY_WINDOW_MIN
        free_list.sort()

    # ── Density-proximity delay post-pass ────────────────────────────────────
    # For each gate class, count how many flights depart within DENSITY_WINDOW_MIN
    # of each other (concurrent ground presence) and penalise high fill ratios.
    dep_arr = dep_times_min.astype(np.float64)
    density_delays = np.zeros(len(assignments), dtype=np.float64)

    for g_idx in range(NUM_GATES):
        g_mask    = (assignments == g_idx)
        if not g_mask.any():
            continue
        g_indices = np.where(g_mask)[0]
        g_times   = dep_arr[g_indices]                  # [Ng]
        total_slots = float(GATE_SLOTS[GATE_CLASSES[g_idx]])

        # Pairwise time differences: dt[i, j] = |t_i - t_j|
        dt = np.abs(g_times[:, None] - g_times[None, :])  # [Ng, Ng]
        concurrent = (dt < DENSITY_WINDOW_MIN).sum(axis=1).astype(np.float64)  # [Ng]

        fill_ratio = concurrent / total_slots             # [Ng]
        excess     = np.maximum(0.0, fill_ratio - DENSITY_THRESHOLD)
        density_delays[g_indices] = DENSITY_FACTOR * excess ** 2

    wait_times += density_delays
    return wait_times


# ===========================================================================
class GreedyFirstFit:
    """
    Deterministic greedy gate-assignment baseline (π₀).

    Assigns each flight to the first authorised gate class in priority order.
    Widebody aircraft are automatically restricted to Wide gate classes.

    Parameters
    ----------
    gate_mapping_path : str or Path
        Path to gate_mapping.json.
    """

    def __init__(self, gate_mapping_path: str = str(GATE_MAP)):
        with open(gate_mapping_path) as fh:
            raw = json.load(fh)

        # valid_gates[(carrier, airport)] → list of gate class indices in priority order
        self.valid_gates: dict = {}

        for airport, terminals in raw.items():
            if airport not in AIRPORT_GATE_ORDER:
                continue
            for terminal_name, carriers in terminals.items():
                gate_indices = _TERMINAL_TO_GATE_INDICES.get(terminal_name, [])
                for carrier in carriers:
                    key = (carrier, airport)
                    self.valid_gates.setdefault(key, [])
                    for g_idx in gate_indices:
                        if g_idx not in self.valid_gates[key]:
                            self.valid_gates[key].append(g_idx)

        # Sort each carrier's valid gates into the greedy priority order
        for key in self.valid_gates:
            airport  = key[1]
            priority = AIRPORT_GATE_ORDER[airport]
            self.valid_gates[key] = sorted(
                self.valid_gates[key], key=lambda g: priority.index(g)
            )

    # -----------------------------------------------------------------------
    def _resolve_airport(self, row) -> str:
        """Return 'EWR' if the flight touches Newark, else None."""
        if row["ORIGIN"] == "EWR":
            return "EWR"
        if row.get("DEST") == "EWR":
            return "EWR"
        return None

    # -----------------------------------------------------------------------
    def assign(self, df: pd.DataFrame) -> np.ndarray:
        """
        Assign a gate class index to every flight.

        Widebody aircraft are filtered to Wide gate classes only.

        Parameters
        ----------
        df : DataFrame
            Must contain at minimum: FL_DATE, CRS_DEP_TIME,
            OP_UNIQUE_CARRIER, ORIGIN, WIDEBODY.

        Returns
        -------
        assignments : ndarray[int], shape (N,)
            Gate class index 0-4 per flight, or UNASSIGNED (-1).
        """
        sort_idx  = np.lexsort((df["CRS_DEP_TIME"].values, df["FL_DATE"].values))
        sorted_df = df.iloc[sort_idx].reset_index(drop=True)

        assignments = np.full(len(sorted_df), UNASSIGNED, dtype=np.int8)

        widebody_col = sorted_df.get("WIDEBODY", pd.Series(0, index=sorted_df.index))

        for i, row in sorted_df.iterrows():
            airport = self._resolve_airport(row)
            if airport is None:
                continue

            carrier   = row["OP_UNIQUE_CARRIER"]
            is_wide   = bool(widebody_col.iloc[i])
            key       = (carrier, airport)
            candidates = self.valid_gates.get(key, [])

            if candidates:
                if is_wide:
                    # Widebody: only Wide gate classes (idx 0 or 3)
                    wide_candidates = [g for g in candidates if not IS_NARROW_GATE[GATE_CLASSES[g]]]
                    chosen = wide_candidates[0] if wide_candidates else FALLBACK_GATE[airport]
                else:
                    chosen = candidates[0]
            else:
                # Unmapped carrier — fallback
                chosen = 0 if is_wide else FALLBACK_GATE[airport]  # A_Wide or A_Narrow

            assignments[i] = chosen

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

        Returns
        -------
        dict with keys:
            f1_violations, f1_rate,
            f2_mean_dist_m, f2_per_gate_class,
            f3_taxi_min, f3_queue_min, f3_total_min,
            f3_mean_pos_delay_min, terminal_counts, n_unassigned
        """
        N = len(df)
        assigned_mask = assignments != UNASSIGNED

        widebody_arr = df.get("WIDEBODY", pd.Series(0, index=df.index)).fillna(0).values

        # ── F1: gate-constraint violations ──────────────────────────────────────
        violations = 0
        for i, row in df.iterrows():
            a = assignments[i]
            if a == UNASSIGNED:
                continue
            airport  = self._resolve_airport(row)
            carrier  = row["OP_UNIQUE_CARRIER"]
            key      = (carrier, airport)
            valid    = self.valid_gates.get(key, [])
            is_wide  = bool(widebody_arr[i])
            # Check carrier auth violation
            if valid and a not in valid:
                violations += 1
            # Check widebody using narrow gate
            elif is_wide and IS_NARROW_GATE[GATE_CLASSES[a]]:
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

        f2_per_gate_class = {}
        for gate_name, dist_m in APPROX_DIST_M.items():
            idx   = GATE_TO_IDX[gate_name]
            count = int((assignments == idx).sum())
            f2_per_gate_class[gate_name] = {"count": count, "dist_m": dist_m}

        # ── F3: physics-based slot-aware queueing simulation ─────────────────
        fl_dates_min = (
            pd.to_datetime(df["FL_DATE"]).astype(np.int64) // (10 ** 6 * 60)
        ).values
        dep_min_day = (
            (df["CRS_DEP_TIME"].values // 100) * 60
            + df["CRS_DEP_TIME"].values % 100
        )
        dep_times = (fl_dates_min + dep_min_day).astype(float)

        wait = simulate_queuing_f3(assignments, dep_times, widebody_arr)

        if assigned_mask.any():
            taxi_min_arr = np.array(
                [APPROX_DIST_M[GATE_CLASSES[int(a)]] / TAXI_SPEED_M_PER_MIN
                 for a in assignments[assigned_mask]],
                dtype=np.float64,
            )
            f3_taxi_min  = float(taxi_min_arr.mean())
            f3_queue_min = float(wait[assigned_mask].mean())
            f3_total_min = float((taxi_min_arr + wait[assigned_mask]).mean())
        else:
            f3_taxi_min = f3_queue_min = f3_total_min = 0.0

        # Raw DEP_DELAY reference (historical label — not assignment-dependent)
        delays = pd.to_numeric(df.get("DEP_DELAY", pd.Series()), errors="coerce").fillna(0)
        f3_mean_pos_delay_min = float(delays.clip(lower=0).mean())

        # Gate class distribution
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
            # F2 (taxi distance)
            f2_mean_dist_m  = f2_mean,
            f2_per_gate_class = f2_per_gate_class,
            # F3 — physics-based slot-aware queueing (primary metrics)
            f3_taxi_min           = f3_taxi_min,
            f3_queue_min          = f3_queue_min,
            f3_total_min          = f3_total_min,
            # F3 — raw DEP_DELAY reference (historical; not assignment-dependent)
            f3_mean_pos_delay_min = f3_mean_pos_delay_min,
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
    sep = "─" * 68
    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")
    print(f"  Flights evaluated  : {metrics['n_flights']:>10,}")
    print(f"  Assigned           : {metrics['n_assigned']:>10,}")
    print(f"  Unassigned (no NYC): {metrics['n_unassigned']:>10,}")
    print(sep)

    # F1
    print(f"  F1  Gate violations     : {metrics['f1_violations']:>6,}  "
          f"({100 * metrics['f1_rate']:.3f}%)")

    # F2
    print(f"  F2  Mean taxi distance  : {metrics['f2_mean_dist_m']:>8.0f} m")
    gc = metrics.get("f2_per_gate_class", {})
    for gate_name in GATE_CLASSES:
        if gate_name in gc:
            info = gc[gate_name]
            print(f"       {gate_name:<18}: {info['count']:>8,} flights  ({info['dist_m']} m)")

    # F3 — queueing simulation
    print(sep)
    print(f"  F3  Taxi time (physics)     : {metrics['f3_taxi_min']:>7.2f} min")
    print(f"  F3  Gate queue wait         : {metrics['f3_queue_min']:>7.2f} min")
    print(f"  F3  Total simulated delay   : {metrics['f3_total_min']:>7.2f} min")
    print(f"      DEP_DELAY ref (hist.)   : {metrics['f3_mean_pos_delay_min']:>7.2f} min")
    print(f"{'=' * 68}\n")


# ===========================================================================
# Standalone entry point
# ===========================================================================
if __name__ == "__main__":
    print("Loading flight data…")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    print(f"  {len(df):,} flights loaded.\n")

    policy           = GreedyFirstFit(GATE_MAP)
    assignments, metrics = policy.run(df)

    print_report(metrics)

    out_path = ROOT / "outputs" / "baseline_assignments.csv"
    out_path.parent.mkdir(exist_ok=True)
    pd.DataFrame({
        "flight_idx"  : np.arange(len(df)),
        "assignment"  : assignments,
        "gate_name"   : [IDX_TO_GATE.get(a, "UNASSIGNED") for a in assignments],
    }).to_csv(out_path, index=False)
    print(f"Assignments saved → {out_path}")
