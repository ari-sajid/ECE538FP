"""
Safe Override Policy (π*).

Combines the operational baseline (GreedyCapacityAware) with learned GNN
proposals (πθ) using two practical safety gates:

    Accept GNN proposal for flight i  iff:
      (1) Confidence: max gate probability ≥ conf_threshold
      (2) Capacity:   gate class not overloaded in its 30-min departure window

    Otherwise: revert to baseline (low-confidence) or repair to next-best
               authorised gate class (capacity overflow).

Aircraft-type feasibility is enforced: widebody flights cannot be placed
on Narrow gate classes, even during repair.

Standalone comparison
---------------------
    python src/safe_override.py [--checkpoint outputs/best_model.pt]
                                [--conf_threshold 0.5]
                                [--split test]   # 'train' | 'test' | 'all'
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
RAW_CSV  = ROOT / "data" / "raw"  / "nyc_master_2025.csv"
GATE_MAP = ROOT / "data" / "meta" / "gate_mapping.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import (                          # noqa: E402
    GreedyFirstFit, print_report,
    GATE_CLASSES, GATE_TO_IDX, IDX_TO_GATE,
    NUM_GATES, UNASSIGNED, APPROX_DIST_M,
    GATE_SLOTS, IS_NARROW_GATE,
)


# ===========================================================================
class SafeOverridePolicy:
    """
    π* = confidence-filtered, capacity-repaired GNN with baseline fallback.

    Parameters
    ----------
    gate_mapping_path : str or Path
    conf_threshold : float
        Minimum gate probability for a GNN proposal to be accepted.
    cap_per_window : int or None
        Maximum slot-units per (gate class, 30-min bucket).
        If None, limits are derived from GATE_SLOTS (total slots per class).
    """

    def __init__(
        self,
        gate_mapping_path: str = str(GATE_MAP),
        conf_threshold: float = 0.3,
        cap_per_window: int = None,
    ):
        self.conf_threshold = conf_threshold
        # Per-gate-class slot capacity for the 30-min window cap.
        # Use total GATE_SLOTS as the cap (generous; prevents extreme overload).
        self._gate_cap: dict = {
            g: GATE_SLOTS[GATE_CLASSES[g]]
            for g in range(NUM_GATES)
        }
        if cap_per_window is not None:
            self._gate_cap = {g: cap_per_window for g in self._gate_cap}

        self.baseline = GreedyFirstFit(gate_mapping_path)
        # Feasibility lookup: (carrier, airport) → set of valid gate indices
        self._valid: dict = {k: set(v) for k, v in self.baseline.valid_gates.items()}

    # -----------------------------------------------------------------------
    def _is_feasible(self, gate_idx: int, carrier: str, airport: str,
                     is_widebody: bool = False) -> bool:
        """Return True if gate_idx is authorised for this carrier/aircraft type."""
        if airport != "EWR":
            return False
        # Widebody cannot use narrow gate classes
        if is_widebody and IS_NARROW_GATE[GATE_CLASSES[gate_idx]]:
            return False
        key   = (carrier, airport)
        valid = self._valid.get(key)
        if valid is None:
            return True   # unmapped carrier → fallback always feasible
        return gate_idx in valid

    # -----------------------------------------------------------------------
    def apply(
        self,
        gate_probs: np.ndarray,            # [N, G] probabilities (already masked)
        baseline_assignments: np.ndarray,  # [N]
        carriers: np.ndarray,              # [N] carrier code per flight
        airports: np.ndarray,              # [N] 'EWR' | None per flight
        dep_times: np.ndarray = None,      # [N] minutes-since-epoch for bucketing
        widebody_arr: np.ndarray = None,   # [N] 1 = widebody, 0 = narrowbody
        window_min: int = 30,
    ) -> tuple:
        """
        Compute the final safe assignment π* and a per-flight decision log.

        Returns
        -------
        final_assignments : ndarray[int]
        decision_log      : DataFrame
        stats             : dict
        """
        N = len(baseline_assignments)
        final  = baseline_assignments.copy()
        reason = np.full(N, "baseline_default", dtype=object)

        if widebody_arr is None:
            widebody_arr = np.zeros(N, dtype=np.float32)

        # ── Step 1: argmax of masked probs is always feasible ────────────────
        gnn_assignments = gate_probs.argmax(axis=-1).astype(np.int8)
        max_prob        = gate_probs.max(axis=-1)

        # ── Step 2: confidence gate ──────────────────────────────────────────
        is_ewr    = np.array([a == "EWR" for a in airports], dtype=bool)
        confident = (max_prob >= self.conf_threshold) & is_ewr

        final[confident]  = gnn_assignments[confident]
        reason[confident] = "accepted_gnn"
        reason[~confident & is_ewr & (gnn_assignments != UNASSIGNED)] = "low_confidence"

        # ── Step 3: capacity repair ──────────────────────────────────────────
        if dep_times is not None:
            buckets = (dep_times.astype(int) // window_min)

            # Count accepted GNN load per (gate, bucket) in slot-units
            load: dict = defaultdict(int)
            for i in range(N):
                if reason[i] == "accepted_gnn":
                    slots = 2 if widebody_arr[i] else 1
                    load[(int(final[i]), int(buckets[i]))] += slots

            overloaded = {
                k: v for k, v in load.items()
                if v > self._gate_cap[k[0]]
            }

            for (g, b), cnt in overloaded.items():
                cap = self._gate_cap[g]
                in_cell = [
                    i for i in range(N)
                    if reason[i] == "accepted_gnn"
                    and int(final[i]) == g
                    and int(buckets[i]) == b
                ]
                in_cell.sort(key=lambda i: max_prob[i])  # least confident first

                # Remove flights until slot-units ≤ cap
                excess_units = cnt - cap
                for i in in_cell:
                    if excess_units <= 0:
                        break
                    slots_i = 2 if widebody_arr[i] else 1
                    is_wide = bool(widebody_arr[i])
                    # Try next-best authorised gate by descending probability
                    sorted_gates = np.argsort(gate_probs[i])[::-1]
                    repaired = False
                    for alt_g in sorted_gates:
                        if alt_g == g:
                            continue
                        if self._is_feasible(int(alt_g), str(carriers[i]),
                                             str(airports[i]), is_wide):
                            final[i]  = alt_g
                            reason[i] = "capacity_repaired"
                            load[(int(alt_g), b)] += slots_i
                            load[(g, b)]          -= slots_i
                            excess_units          -= slots_i
                            repaired = True
                            break
                    if not repaired:
                        final[i]  = baseline_assignments[i]
                        reason[i] = "capacity_repaired_fallback"
                        excess_units -= slots_i

        stats = dict(
            n_total                      = N,
            n_accepted_gnn               = int((reason == "accepted_gnn").sum()),
            n_low_confidence             = int((reason == "low_confidence").sum()),
            n_capacity_repaired          = int((reason == "capacity_repaired").sum()),
            n_capacity_repaired_fallback = int((reason == "capacity_repaired_fallback").sum()),
            n_baseline_default           = int((reason == "baseline_default").sum()),
            conf_threshold               = self.conf_threshold,
            gate_caps                    = self._gate_cap,
            mean_confidence              = float(max_prob.mean()),
        )

        decision_log = pd.DataFrame({
            "flight_idx"         : np.arange(N),
            "baseline_assignment": baseline_assignments,
            "gnn_assignment"     : gnn_assignments,
            "final_assignment"   : final,
            "max_gate_prob"      : max_prob,
            "carrier"            : carriers,
            "airport"            : airports,
            "decision"           : reason,
        })

        return final, decision_log, stats

    # -----------------------------------------------------------------------
    def score(self, assignments: np.ndarray, df: pd.DataFrame) -> dict:
        """Delegate to GreedyFirstFit.score() — same metric computation."""
        return self.baseline.score(assignments, df)


# ===========================================================================
# Standalone entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Run safe override policy and compare with baseline"
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to best_model.pt (optional)")
    p.add_argument("--conf_threshold", type=float, default=0.5)
    p.add_argument("--cap_per_window", type=int, default=None)
    p.add_argument("--split", choices=["train", "test", "all"], default="test")
    return p.parse_args()


def main():
    args = parse_args()
    print("Safe Override — decision log inspector")
    print("=" * 42)
    print("Full policy comparison is generated automatically at the end of:")
    print("    python -m src.train")
    print("Outputs: outputs/policy_comparison.csv  +  safe_override_decisions.csv")
    print()

    if not (args.checkpoint and Path(args.checkpoint).exists()):
        if args.checkpoint:
            print(f"Checkpoint not found: {args.checkpoint}")
        print("Run training first to generate policy comparison:")
        print("    python -m src.train --epochs 50")
        return

    print("Loading flight data...")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

    if args.split == "train":
        df = df[df["FL_DATE"].dt.month <= 10].reset_index(drop=True)
    elif args.split == "test":
        df = df[df["FL_DATE"].dt.month  > 10].reset_index(drop=True)
    print(f"  {len(df):,} flights ({args.split} split).\n")

    from src.baselines import GreedyCapacityAware  # noqa: E402
    policy               = SafeOverridePolicy(
        conf_threshold=args.conf_threshold,
        cap_per_window=args.cap_per_window,
    )
    baseline_assignments = GreedyCapacityAware(str(GATE_MAP)).assign(df)
    baseline_metrics     = policy.baseline.score(baseline_assignments, df)
    print_report(baseline_metrics, title="GreedyCapacityAware Baseline")

    dec_path = ROOT / "outputs" / "safe_override_decisions.csv"
    if dec_path.exists():
        dec = pd.read_csv(dec_path)
        print("\nDecision summary from last training run:")
        print(dec["decision"].value_counts().to_string())
        print(f"\nMean gate confidence: {dec['max_gate_prob'].mean():.3f}")
    else:
        print("\nNo decision log found. Run training to generate one.")


if __name__ == "__main__":
    main()
