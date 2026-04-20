"""
Safe Override Policy (π*).

Combines the deterministic baseline (π₀) with learned GNN proposals (πθ)
using two practical safety gates:

    Accept GNN proposal for flight i  iff:
      (1) Confidence: max gate probability ≥ conf_threshold
      (2) Capacity:   terminal not overloaded in its 30-min departure window

    Otherwise: revert to π₀ (low-confidence) or repair to next-best
               authorised terminal (capacity overflow).

Gate feasibility is enforced structurally upstream (GateMasker.mask_logits
→ −∞ before softmax), so argmax of gate_probs is always a valid terminal.
No per-flight feasibility check is needed here.

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
)
from src.loss import TERMINAL_CAPACITY              # noqa: E402


# ===========================================================================
class SafeOverridePolicy:
    """
    π* = confidence-filtered, capacity-repaired GNN with π₀ fallback.

    Parameters
    ----------
    gate_mapping_path : str or Path
    conf_threshold : float
        Minimum gate probability for a GNN proposal to be accepted.
        Flights below this threshold fall back to the baseline.
        Default: 0.5.
    cap_per_window : int or None
        Maximum GNN-accepted flights per (terminal, 30-min bucket).
        If None, limits are derived from TERMINAL_CAPACITY (total gate count).
        Excess flights are reassigned to the next-best authorised terminal.
    """

    def __init__(
        self,
        gate_mapping_path: str = str(GATE_MAP),
        conf_threshold: float = 0.3,
        cap_per_window: int = None,
    ):
        self.conf_threshold = conf_threshold
        # Build per-terminal capacity: TERMINAL_CAPACITY gates // 4 per 30-min slot
        self._terminal_cap: dict = {
            g: TERMINAL_CAPACITY[GATE_CLASSES[g]]
            for g in range(len(GATE_CLASSES))
        }
        # Allow override with a single flat cap
        if cap_per_window is not None:
            self._terminal_cap = {g: cap_per_window for g in self._terminal_cap}
        self.baseline = GreedyFirstFit(gate_mapping_path)

        # Feasibility lookup: (carrier, airport) → set of valid gate indices
        self._valid: dict = {k: set(v) for k, v in self.baseline.valid_gates.items()}

    # -----------------------------------------------------------------------
    def _is_feasible(self, gate_idx: int, carrier: str, airport: str) -> bool:
        """Return True if gate_idx is authorised for this carrier at this airport."""
        if airport not in ("EWR", "LGA"):
            return False
        key = (carrier, airport)
        valid = self._valid.get(key)
        if valid is None:
            return True   # unmapped carrier → fallback terminal always feasible
        return gate_idx in valid

    # -----------------------------------------------------------------------
    def apply(
        self,
        gate_probs: np.ndarray,            # [N, G] probabilities (already masked)
        baseline_assignments: np.ndarray,  # [N]
        carriers: np.ndarray,              # [N] carrier code per flight
        airports: np.ndarray,              # [N] 'EWR' | 'LGA' | None per flight
        dep_times: np.ndarray = None,      # [N] CRS_DEP_TIME for capacity bucketing
        window_min: int = 30,
    ) -> tuple:
        """
        Compute the final safe assignment π* and a per-flight decision log.

        Parameters
        ----------
        gate_probs          : ndarray[float32, shape (N, G)]
                              Gate probabilities after GateMasker; argmax is
                              always a feasible terminal by construction.
        baseline_assignments: ndarray[int]  – output of GreedyFirstFit.assign()
        carriers            : ndarray[str]
        airports            : ndarray[str | None]
        dep_times           : ndarray[int]  – minutes-since-epoch (date + time)
                              for capacity bucketing; pass date-aware values to
                              avoid cross-day bucket collisions.  Skip capacity
                              repair if None.
        window_min          : int  – bucket width in minutes (default 30)

        Returns
        -------
        final_assignments : ndarray[int]   – π* gate indices
        decision_log      : DataFrame      – per-flight accept/reject with reason
        stats             : dict           – summary counts
        """
        N = len(baseline_assignments)
        final  = baseline_assignments.copy()
        reason = np.full(N, "baseline_default", dtype=object)

        # ── Step 1: argmax of masked probs is always feasible ────────────────
        gnn_assignments = gate_probs.argmax(axis=-1).astype(np.int8)
        max_prob        = gate_probs.max(axis=-1)

        # ── Step 2: confidence gate ──────────────────────────────────────────
        # Only override flights at EWR/LGA; non-airport flights stay on baseline.
        is_nyc = np.array([a in ("EWR", "LGA") for a in airports], dtype=bool)
        confident = (max_prob >= self.conf_threshold) & is_nyc

        final[confident]  = gnn_assignments[confident]
        reason[confident] = "accepted_gnn"
        reason[~confident & is_nyc & (gnn_assignments != UNASSIGNED)] = "low_confidence"

        # ── Step 3: capacity repair ──────────────────────────────────────────
        if dep_times is not None:
            buckets = (dep_times.astype(int) // window_min)

            # Count accepted GNN load per (gate, bucket)
            load: dict = defaultdict(int)
            for i in range(N):
                if reason[i] == "accepted_gnn":
                    load[(int(final[i]), int(buckets[i]))] += 1

            # Identify overloaded (gate, bucket) pairs using per-terminal cap
            overloaded = {
                k: v for k, v in load.items()
                if v > self._terminal_cap[k[0]]
            }

            for (g, b), cnt in overloaded.items():
                cap = self._terminal_cap[g]
                # Collect flights in this overloaded cell, sorted by confidence ↑
                in_cell = [
                    i for i in range(N)
                    if reason[i] == "accepted_gnn"
                    and int(final[i]) == g
                    and int(buckets[i]) == b
                ]
                in_cell.sort(key=lambda i: max_prob[i])  # least confident first

                excess = in_cell[: cnt - cap]
                for i in excess:
                    # Try next-best authorised gate by descending probability
                    sorted_gates = np.argsort(gate_probs[i])[::-1]
                    repaired = False
                    for alt_g in sorted_gates:
                        if alt_g == g:
                            continue
                        if self._is_feasible(int(alt_g), str(carriers[i]), str(airports[i])):
                            final[i]  = alt_g
                            reason[i] = "capacity_repaired"
                            load[(int(alt_g), b)] += 1
                            load[(g, b)]          -= 1
                            repaired = True
                            break
                    if not repaired:
                        final[i]  = baseline_assignments[i]
                        reason[i] = "capacity_repaired_fallback"

        stats = dict(
            n_total                    = N,
            n_accepted_gnn             = int((reason == "accepted_gnn").sum()),
            n_low_confidence           = int((reason == "low_confidence").sum()),
            n_capacity_repaired        = int((reason == "capacity_repaired").sum()),
            n_capacity_repaired_fallback = int((reason == "capacity_repaired_fallback").sum()),
            n_baseline_default         = int((reason == "baseline_default").sum()),
            conf_threshold             = self.conf_threshold,
            terminal_caps              = self._terminal_cap,
            mean_confidence            = float(max_prob.mean()),
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
# Comparison table
# ===========================================================================

def print_comparison(
    baseline_metrics: dict,
    override_metrics: dict,
    override_stats: dict,
):
    """Print a side-by-side F2 / F3 comparison of π₀ vs π*."""
    sep = "─" * 70

    def pct_change(new, old):
        if old == 0:
            return "  n/a  "
        delta = (new - old) / abs(old) * 100
        sign  = "+" if delta >= 0 else ""
        return f"{sign}{delta:+.1f}%"

    print(f"\n{'=' * 70}")
    print(f"  Policy Comparison: π₀ (Baseline) vs π* (Safe Override)")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<35} {'π₀ Baseline':>12}  {'π* Override':>12}  {'Δ':>8}")
    print(sep)

    # F2
    bd = baseline_metrics["f2_mean_dist_m"]
    od = override_metrics["f2_mean_dist_m"]
    print(f"  {'F2  Mean taxi distance (m)':<35} {bd:>12.0f}  {od:>12.0f}  {pct_change(od, bd):>8}")

    # F3
    bp = baseline_metrics["f3_mean_pos_delay_min"]
    op = override_metrics["f3_mean_pos_delay_min"]
    print(f"  {'F3  Mean positive delay (min)':<35} {bp:>12.2f}  {op:>12.2f}  {pct_change(op, bp):>8}")

    print(sep)
    n = override_stats["n_total"]
    print(f"  Flights evaluated                  : {n:,}")
    print(f"  Accepted (confident GNN)           : "
          f"{override_stats['n_accepted_gnn']:,}  "
          f"({100 * override_stats['n_accepted_gnn'] / max(n, 1):.1f}%)")
    print(f"  Fallback (low confidence < {override_stats['conf_threshold']:.2f})   : "
          f"{override_stats['n_low_confidence']:,}  "
          f"({100 * override_stats['n_low_confidence'] / max(n, 1):.1f}%)")
    print(f"  Capacity repaired (alt terminal)   : "
          f"{override_stats['n_capacity_repaired']:,}  "
          f"({100 * override_stats['n_capacity_repaired'] / max(n, 1):.1f}%)")
    print(f"  Capacity repaired (→ baseline)     : "
          f"{override_stats['n_capacity_repaired_fallback']:,}  "
          f"({100 * override_stats['n_capacity_repaired_fallback'] / max(n, 1):.1f}%)")
    print(f"  Mean gate confidence               : {override_stats['mean_confidence']:.3f}")
    caps = override_stats['terminal_caps']
    cap_str = ", ".join(f"{GATE_CLASSES[g].split('_',1)[1]}:{v}"
                        for g, v in caps.items())
    print(f"  Capacity caps (gates//4 per 30 min): {cap_str}")
    print(f"{'=' * 70}\n")


# ===========================================================================
# Standalone entry point — loads data + optional GNN checkpoint
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Run safe override policy and compare with baseline"
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to best_model.pt (optional; random probs used if absent)")
    p.add_argument("--conf_threshold", type=float, default=0.5,
                   help="Min gate probability to accept GNN proposal (default 0.5)")
    p.add_argument("--cap_per_window", type=int, default=None,
                   help="Max GNN assignments per (terminal, 30-min window) "
                        "(default: TERMINAL_CAPACITY//4 per terminal)")
    p.add_argument("--split", choices=["train", "test", "all"], default="test",
                   help="Which months to evaluate on (default: test = months 11-12)")
    return p.parse_args()


def main():
    args = parse_args()

    # Safe Override requires real GNN gate probabilities, which can only be
    # produced by running the trained model over the full HeteroData graph.
    # Policy comparison (pi0 vs pi*) is generated automatically inside train.py.
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

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading flight data...")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

    if args.split == "train":
        df = df[df["FL_DATE"].dt.month <= 10].reset_index(drop=True)
    elif args.split == "test":
        df = df[df["FL_DATE"].dt.month  > 10].reset_index(drop=True)
    print(f"  {len(df):,} flights ({args.split} split).\n")

    # ── Show baseline metrics and last decision log ────────────────────────
    print("Running baseline for reference...")
    policy               = SafeOverridePolicy(
        conf_threshold=args.conf_threshold,
        cap_per_window=args.cap_per_window,
    )
    baseline_assignments = policy.baseline.assign(df)
    baseline_metrics     = policy.baseline.score(baseline_assignments, df)
    print_report(baseline_metrics, title="Greedy First-Fit Baseline")

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
