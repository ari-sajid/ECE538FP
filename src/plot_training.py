"""
Training metrics visualization for ECE538FP GNN.

Reads:
  outputs/training_history.csv   — per-epoch train/val F2 and F3
  outputs/pareto_front.csv       — Pareto-optimal epochs (val F2 vs F3)
  outputs/baseline_comparison.csv — greedy baseline scores

Produces outputs/training_metrics.png with five panels:
  A. F2 (taxi distance) — train vs val, with baseline reference lines
  B. F3 (delay risk)    — train vs val
  C. Pareto front       — val F2 vs val F3, coloured by epoch
  D. Generalisation gap — val minus train for F2 and F3
  E. Baseline comparison bar chart

Usage:
    python src/plot_training.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
HIST_CSV = ROOT / "outputs" / "training_history.csv"
PARE_CSV = ROOT / "outputs" / "pareto_front.csv"
BASE_CSV = ROOT / "outputs" / "baseline_comparison.csv"
OUT_PATH = ROOT / "outputs" / "training_metrics.png"

# Max taxi distance used to normalise F2 in the GNN (EWR_Terminal_C)
F2_MAX_M = 1380.0


def main():
    if not HIST_CSV.exists():
        print(f"ERROR: {HIST_CSV} not found. Run training first.")
        return

    hist   = pd.read_csv(HIST_CSV)
    epochs = hist["epoch"].values

    # Convert normalised F2 → metres for interpretable y-axis
    hist["train_F2_m"] = hist["train_F2"] * F2_MAX_M
    hist["val_F2_m"]   = hist["val_F2"]   * F2_MAX_M

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#f8f9fa")
    fig.suptitle(
        "GNN Training Metrics — ECE538FP\n"
        "Spatio-Temporal GNN for Airport Gate Scheduling (EWR & LGA)",
        fontsize=13, fontweight="bold", y=1.01, color="#1a1a2e",
    )

    # ── Panel A: F2 (taxi distance in metres) ──────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, hist["train_F2_m"], color="#2ca02c", lw=2, label="Train F2")
    ax.plot(epochs, hist["val_F2_m"],   color="#2ca02c", lw=2, ls="--",
            alpha=0.75, label="Val F2")
    ax.fill_between(epochs, hist["train_F2_m"], hist["val_F2_m"],
                    alpha=0.08, color="#2ca02c")

    # Baseline reference lines
    if BASE_CSV.exists():
        base = pd.read_csv(BASE_CSV)
        colours_b = ["#e41a1c", "#984ea3", "#ff7f00"]
        for (_, row), col in zip(base.iterrows(), colours_b):
            ax.axhline(row["f2_mean_dist_m"], color=col, lw=1.2, ls=":",
                       label=f"{row['policy'].strip()} ({row['f2_mean_dist_m']:.0f} m)")

    ax.set_title("F2 – Mean Taxi Distance (m)\nGNN vs Greedy Baselines",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Mean taxi distance (m)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=8)

    # ── Panel B: F3 (delay risk probability) ───────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, hist["train_F3"], color="#ff7f0e", lw=2, label="Train F3")
    ax.plot(epochs, hist["val_F3"],   color="#ff7f0e", lw=2, ls="--",
            alpha=0.75, label="Val F3")
    ax.fill_between(epochs, hist["train_F3"], hist["val_F3"],
                    alpha=0.08, color="#ff7f0e")

    if BASE_CSV.exists():
        base = pd.read_csv(BASE_CSV)
        # baseline F3 delay risk is stored as percent — convert to fraction
        base_f3 = base["f3_delay_risk_pct"].iloc[0] / 100.0
        ax.axhline(base_f3, color="#e41a1c", lw=1.2, ls=":",
                   label=f"Baseline risk ({base_f3:.3f})")

    ax.set_title("F3 – Delay Risk  P(delay > 15 min)\n[sigmoid proxy]",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("P(delay > 15 min)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=8)

    # ── Panel C: F2 + F3 combined on dual axis ──────────────────────────────
    ax = axes[0, 2]
    ax2 = ax.twinx()
    l1, = ax.plot(epochs, hist["val_F2_m"],  color="#2ca02c", lw=2, label="Val F2 (m)")
    l2, = ax2.plot(epochs, hist["val_F3"],   color="#ff7f0e", lw=2, ls="--", label="Val F3")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("F2 taxi distance (m)", color="#2ca02c", fontsize=9)
    ax2.set_ylabel("F3 delay risk", color="#ff7f0e", fontsize=9)
    ax.tick_params(axis="y", labelcolor="#2ca02c", labelsize=8)
    ax2.tick_params(axis="y", labelcolor="#ff7f0e", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Val F2 vs Val F3 Over Training\n(dual axis)", fontweight="bold", fontsize=10)
    ax.legend(handles=[l1, l2], fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, lw=0.5)

    # ── Panel D: Pareto front ───────────────────────────────────────────────
    ax = axes[1, 0]
    if PARE_CSV.exists():
        pareto = pd.read_csv(PARE_CSV)
        sc = ax.scatter(
            pareto["val_F2"] * F2_MAX_M, pareto["val_F3"],
            c=pareto["epoch"], cmap="plasma",
            s=120, edgecolors="black", linewidths=0.6, zorder=3,
        )
        for _, row in pareto.iterrows():
            ax.annotate(
                f"ep{int(row['epoch'])}",
                (row["val_F2"] * F2_MAX_M, row["val_F3"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7,
            )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Epoch", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "pareto_front.csv not found",
                transform=ax.transAxes, ha="center", fontsize=9)
    ax.set_xlabel("F2 – Taxi Distance (m)", fontsize=9)
    ax.set_ylabel("F3 – Delay Risk", fontsize=9)
    ax.set_title("Pareto-Optimal Val Epochs\n(lower-left is better for both)",
                 fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=8)

    # ── Panel E: Generalisation gap ─────────────────────────────────────────
    ax = axes[1, 1]
    gap_f2 = (hist["val_F2_m"] - hist["train_F2_m"])
    gap_f3 = (hist["val_F3"]   - hist["train_F3"]) * 100   # scale for visibility
    bar_w  = 0.4
    ax.bar(epochs - bar_w/2, gap_f2, width=bar_w, color="#2ca02c", alpha=0.75, label="F2 gap (m)")
    ax.bar(epochs + bar_w/2, gap_f3, width=bar_w, color="#ff7f0e", alpha=0.75, label="F3 gap x100")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Generalisation Gap (Val - Train)\nPositive = val worse than train",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Gap", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, lw=0.5, axis="y")
    ax.tick_params(labelsize=8)

    # ── Panel F: Baseline bar chart ─────────────────────────────────────────
    ax = axes[1, 2]
    if BASE_CSV.exists():
        base     = pd.read_csv(BASE_CSV)
        policies = [p.strip() for p in base["policy"]]
        f2_vals  = base["f2_mean_dist_m"].values
        cv_vals  = base["utilisation_cv"].values

        x      = np.arange(len(policies))
        bar_w  = 0.35
        bars   = ax.bar(x - bar_w/2, f2_vals, bar_w, color="#2ca02c", alpha=0.8, label="F2 dist (m)")
        ax2    = ax.twinx()
        ax2.bar(x + bar_w/2, cv_vals, bar_w, color="#9467bd", alpha=0.8, label="Util CV")

        # Annotate F2 bars
        for bar, val in zip(bars, f2_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=12, ha="right", fontsize=7)
        ax.set_ylabel("Mean taxi distance (m)", color="#2ca02c", fontsize=9)
        ax2.set_ylabel("Utilisation CV", color="#9467bd", fontsize=9)
        ax.tick_params(axis="y", labelcolor="#2ca02c", labelsize=8)
        ax2.tick_params(axis="y", labelcolor="#9467bd", labelsize=8)
        ax.set_title("Greedy Baseline Comparison\n(lower F2 and lower CV is better)",
                     fontweight="bold", fontsize=10)
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3, lw=0.5, axis="y")
    else:
        ax.text(0.5, 0.5, "baseline_comparison.csv not found",
                transform=ax.transAxes, ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
