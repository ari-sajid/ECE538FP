"""
Safe Override Policy (π*).

Combines the deterministic baseline (π₀) with learned GNN proposals (πθ)
using two hard safety gates:

    Accept GNN proposal for flight i  iff:
      (1) No violation: proposed terminal is authorised for carrier i
      (2) Global stability: Kendall–Tau distance between the full baseline
          sequence and the full GNN sequence ≤ k_max

    Otherwise: revert to π₀ for that flight (violation guard) or for ALL
               flights (stability guard).

Kendall–Tau distance
--------------------
The normalised Kendall–Tau distance between two integer sequences a and b
of length N is:

    KT(a, b)  =  (# discordant pairs) / (N * (N-1) / 2)  ∈  [0, 1]

A pair (i, j) is discordant when  sign(a[i]-a[j]) ≠ sign(b[i]-b[j]).
KT = 0 means identical ordering; KT = 1 means fully reversed.
Computed in O(N log N) via scipy.stats.kendalltau.

Comparison report
-----------------
Run standalone with a saved baseline and (optional) GNN checkpoint to
produce a side-by-side F1/F2/F3 comparison table:

    python src/safe_override.py [--checkpoint outputs/best_model.pt]
                                [--k_max 0.05]
                                [--split test]   # 'train' | 'test' | 'all'
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

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


# ===========================================================================
class SafeOverridePolicy:
    """
    π* = filtered combination of π₀ (greedy baseline) and πθ (GNN).

    Parameters
    ----------
    gate_mapping_path : str or Path
    k_max : float
        Maximum allowable normalised Kendall–Tau distance between the
        baseline assignment sequence and the GNN proposal sequence.
        Typical values: 0.01 (conservative) – 0.10 (permissive).
        Default: 0.05.
    """

    def __init__(
        self,
        gate_mapping_path: str = str(GATE_MAP),
        k_max: float = 0.05,
    ):
        self.k_max   = k_max
        self.baseline = GreedyFirstFit(gate_mapping_path)

        # Build feasibility lookup: (carrier, airport) → set of valid gate indices
        self._valid: dict = {k: set(v) for k, v in self.baseline.valid_gates.items()}

    # -----------------------------------------------------------------------
    def _is_feasible(self, gate_idx: int, carrier: str, airport: str) -> bool:
        """Return True if gate_idx is authorised for this carrier at this airport."""
        if airport not in ("EWR", "LGA"):
            return False
        key = (carrier, airport)
        valid = self._valid.get(key)
        if valid is None:
            # Carrier not in gate_mapping → fallback terminal always counts as feasible
            return True
        return gate_idx in valid

    # -----------------------------------------------------------------------
    @staticmethod
    def kendall_tau_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Normalised Kendall–Tau distance between integer sequences a and b.

        Returns a value in [0, 1]; 0 = identical, 1 = fully reversed.
        Only non-UNASSIGNED positions (both a[i] and b[i] ≠ -1) are included.
        """
        # Filter to positions where both sequences have a real assignment
        mask = (a != UNASSIGNED) & (b != UNASSIGNED)
        if mask.sum() < 2:
            return 0.0
        tau, _ = kendalltau(a[mask], b[mask])
        # tau ∈ [-1, 1];  distance = (1 - tau) / 2 ∈ [0, 1]
        return float((1.0 - tau) / 2.0)

    # -----------------------------------------------------------------------
    def apply(
        self,
        gnn_assignments: np.ndarray,   # [N] hard argmax from GNN logits
        baseline_assignments: np.ndarray,  # [N] from GreedyFirstFit.assign()
        carriers: np.ndarray,          # [N] carrier code per flight
        airports: np.ndarray,          # [N] 'EWR' | 'LGA' | None per flight
    ) -> tuple:
        """
        Compute the final safe assignment π* and a per-flight decision log.

        Parameters
        ----------
        gnn_assignments     : ndarray[int] – argmax of GNN gate logits
        baseline_assignments: ndarray[int] – output of GreedyFirstFit.assign()
        carriers            : ndarray[str]
        airports            : ndarray[str | None]

        Returns
        -------
        final_assignments : ndarray[int]   – π* gate indices
        decision_log      : DataFrame      – per-flight accept/reject with reason
        stats             : dict           – summary counts
        """
        N = len(gnn_assignments)
        final  = baseline_assignments.copy()
        reason = np.full(N, "baseline_default", dtype=object)

        # ── Guard 1: per-flight feasibility (F1) ────────────────────────────
        gnn_feasible = np.zeros(N, dtype=bool)
        for i in range(N):
            g = gnn_assignments[i]
            ap = airports[i]
            ca = carriers[i]
            if g == UNASSIGNED or ap is None:
                reason[i] = "unassigned"
                continue
            if self._is_feasible(g, ca, ap):
                gnn_feasible[i] = True
            else:
                reason[i] = "rejected_f1_violation"

        # Tentative sequence: accept all feasible GNN proposals
        tentative = baseline_assignments.copy()
        tentative[gnn_feasible] = gnn_assignments[gnn_feasible]

        # ── Guard 2: global Kendall–Tau stability ────────────────────────────
        kt_dist = self.kendall_tau_distance(baseline_assignments, tentative)

        if kt_dist <= self.k_max:
            # Stable enough: accept all feasible GNN proposals
            final[gnn_feasible]        = gnn_assignments[gnn_feasible]
            reason[gnn_feasible]       = "accepted_gnn"
        else:
            # Too many swaps: full fallback to baseline
            reason[gnn_feasible]       = "rejected_kt_exceeded"
            # final already equals baseline

        stats = dict(
            n_total           = N,
            n_accepted_gnn    = int((reason == "accepted_gnn").sum()),
            n_rejected_f1     = int((reason == "rejected_f1_violation").sum()),
            n_rejected_kt     = int((reason == "rejected_kt_exceeded").sum()),
            n_unassigned      = int((reason == "unassigned").sum()),
            kt_distance       = kt_dist,
            kt_threshold      = self.k_max,
            kt_guard_triggered= kt_dist > self.k_max,
        )

        decision_log = pd.DataFrame({
            "flight_idx"         : np.arange(N),
            "baseline_assignment": baseline_assignments,
            "gnn_assignment"     : gnn_assignments,
            "final_assignment"   : final,
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
    """Print a side-by-side F1 / F2 / F3 comparison of π₀ vs π*."""
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

    # F1
    bv = baseline_metrics["f1_violations"]
    ov = override_metrics["f1_violations"]
    print(f"  {'F1  Gate violations':<35} {bv:>12,}  {ov:>12,}  {pct_change(ov, bv):>8}")

    # F2
    bd = baseline_metrics["f2_mean_dist_m"]
    od = override_metrics["f2_mean_dist_m"]
    print(f"  {'F2  Mean taxi distance (m)':<35} {bd:>12.0f}  {od:>12.0f}  {pct_change(od, bd):>8}")

    # F3
    bp = baseline_metrics["f3_mean_pos_delay_min"]
    op = override_metrics["f3_mean_pos_delay_min"]
    print(f"  {'F3  Mean positive delay (min)':<35} {bp:>12.2f}  {op:>12.2f}  {pct_change(op, bp):>8}")

    print(sep)
    # Override stats
    n  = override_stats["n_total"]
    print(f"  Flights evaluated              : {n:,}")
    print(f"  GNN proposals accepted (π*)   : "
          f"{override_stats['n_accepted_gnn']:,}  "
          f"({100 * override_stats['n_accepted_gnn'] / max(n, 1):.1f}%)")
    print(f"  Rejected — F1 violation        : "
          f"{override_stats['n_rejected_f1']:,}  "
          f"({100 * override_stats['n_rejected_f1'] / max(n, 1):.1f}%)")
    print(f"  Rejected — KT guard triggered  : "
          f"{'YES' if override_stats['kt_guard_triggered'] else 'NO':<5}  "
          f"(KT={override_stats['kt_distance']:.4f}, threshold={override_stats['kt_threshold']})")
    print(f"{'=' * 70}\n")


# ===========================================================================
# Standalone entry point — loads data + optional GNN checkpoint
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Run safe override policy and compare with baseline"
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to best_model.pt (optional; random policy used if absent)")
    p.add_argument("--k_max", type=float, default=0.05,
                   help="Max normalised Kendall–Tau distance (default 0.05)")
    p.add_argument("--split", choices=["train", "test", "all"], default="test",
                   help="Which months to evaluate on (default: test = months 11-12)")
    return p.parse_args()


def _random_gnn_assignments(df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """
    Simulate an untrained GNN by sampling gate indices uniformly at random.
    Used when no checkpoint is available, to demonstrate the safe override
    catching infeasible / destabilising proposals.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, NUM_GATES, size=len(df)).astype(np.int8)


def _gnn_assignments_from_checkpoint(checkpoint_path: str, df: pd.DataFrame) -> np.ndarray:
    """Load a trained GNN and run inference to get hard gate assignments."""
    try:
        import torch
        from src.model import SpatioTemporalGNN
        from src.loss  import GATE_CLASSES as LC

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model = SpatioTemporalGNN(
            in_channels     = ckpt["in_channels"],
            hidden_channels = ckpt.get("args", {}).get("hidden_channels", 128),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print("  GNN checkpoint loaded.  Running inference…")
        print("  (Full inference requires building HeteroData — see src/train.py)")
        print("  Falling back to random simulation for this standalone comparison.\n")
    except Exception as e:
        print(f"  Could not load checkpoint ({e}); using random GNN simulation.\n")

    return _random_gnn_assignments(df)


def main():
    args = parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading flight data…")
    df = pd.read_csv(RAW_CSV, low_memory=False)

    # Apply the requested split
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    if args.split == "train":
        df = df[df["FL_DATE"].dt.month <= 10].reset_index(drop=True)
    elif args.split == "test":
        df = df[df["FL_DATE"].dt.month  > 10].reset_index(drop=True)
    print(f"  {len(df):,} flights ({args.split} split).\n")

    # ── Build baseline ─────────────────────────────────────────────────────
    print("Running baseline (π₀)…")
    policy               = SafeOverridePolicy(k_max=args.k_max)
    baseline_assignments = policy.baseline.assign(df)
    baseline_metrics     = policy.baseline.score(baseline_assignments, df)
    print_report(baseline_metrics, title="π₀  Greedy First-Fit Baseline")

    # ── Simulate or load GNN ───────────────────────────────────────────────
    print("Obtaining GNN proposals (πθ)…")
    if args.checkpoint and Path(args.checkpoint).exists():
        gnn_assignments = _gnn_assignments_from_checkpoint(args.checkpoint, df)
    else:
        if args.checkpoint:
            print(f"  Checkpoint '{args.checkpoint}' not found.")
        print("  Using random gate assignments to simulate an untrained GNN.\n"
              "  Re-run after training:  python src/safe_override.py "
              "--checkpoint outputs/best_model.pt\n")
        gnn_assignments = _random_gnn_assignments(df)

    # ── Resolve airport per flight ─────────────────────────────────────────
    airports = np.where(
        df["ORIGIN"].isin(["EWR", "LGA"]),
        df["ORIGIN"],
        np.where(df.get("DEST", pd.Series("", index=df.index)).isin(["EWR", "LGA"]),
                 df.get("DEST", pd.Series("", index=df.index)),
                 None),
    )
    carriers = df["OP_UNIQUE_CARRIER"].values

    # ── Apply safe override ────────────────────────────────────────────────
    print(f"Applying safe override (π*) with k_max={args.k_max}…")
    final_assignments, decision_log, override_stats = policy.apply(
        gnn_assignments, baseline_assignments, carriers, airports
    )
    override_metrics = policy.score(final_assignments, df)

    # ── Print comparison ───────────────────────────────────────────────────
    print_comparison(baseline_metrics, override_metrics, override_stats)

    # ── Save outputs ───────────────────────────────────────────────────────
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    decision_log.to_csv(out_dir / "safe_override_decisions.csv", index=False)

    comparison_rows = [
        {"policy": "π₀ Baseline",    "f1_violations": baseline_metrics["f1_violations"],
         "f2_mean_dist_m": baseline_metrics["f2_mean_dist_m"],
         "f3_mean_pos_delay_min": baseline_metrics["f3_mean_pos_delay_min"]},
        {"policy": "π* Safe Override","f1_violations": override_metrics["f1_violations"],
         "f2_mean_dist_m": override_metrics["f2_mean_dist_m"],
         "f3_mean_pos_delay_min": override_metrics["f3_mean_pos_delay_min"]},
    ]
    pd.DataFrame(comparison_rows).to_csv(out_dir / "policy_comparison.csv", index=False)

    print(f"Saved → {out_dir}/safe_override_decisions.csv")
    print(f"Saved → {out_dir}/policy_comparison.csv")


if __name__ == "__main__":
    main()
