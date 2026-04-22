"""
Training metrics visualization for ECE538FP GNN.

Reads:
  outputs/training_history.csv   — per-epoch train/val loss, taxi, cong
  outputs/pareto_front.csv       — Pareto-optimal epochs (val_taxi vs val_cong)
  outputs/baseline_comparison.csv — greedy baseline F3 scores

Produces outputs/training_metrics.png with six panels:
  A. Total loss          — train vs val
  B. L_taxi component    — E[T_taxi] train vs val
  C. L_cong component    — soft congestion proxy train vs val
  D. Pareto front        — val_taxi vs val_cong coloured by epoch
  E. Generalisation gap  — val minus train for taxi and cong
  F. Baseline F3 comparison bar chart

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
COMP_CSV = ROOT / "outputs" / "policy_comparison.csv"
OUT_PATH = ROOT / "outputs" / "training_metrics.png"


def main():
    if not HIST_CSV.exists():
        print(f"ERROR: {HIST_CSV} not found. Run training first.")
        return

    hist   = pd.read_csv(HIST_CSV)
    epochs = hist["epoch"].values

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#f8f9fa")
    fig.suptitle(
        "GNN Training Metrics — ECE538FP\n"
        "Spatio-Temporal GNN for Airport Gate Scheduling (EWR & LGA)",
        fontsize=13, fontweight="bold", y=1.01, color="#1a1a2e",
    )

    # ── Panel A: Total loss ────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, hist["train_loss"], color="#1f77b4", lw=2, label="Train loss")
    ax.plot(epochs, hist["val_loss"],   color="#1f77b4", lw=2, ls="--",
            alpha=0.75, label="Val loss")
    ax.fill_between(epochs, hist["train_loss"], hist["val_loss"],
                    alpha=0.08, color="#1f77b4")
    ax.set_title("Total Loss  (β·L_taxi + λ·L_cong + γ·L_turn)",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=8)

    # ── Panel B: L_taxi — expected taxi time ──────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, hist["train_taxi"], color="#2ca02c", lw=2, label="Train L_taxi")
    ax.plot(epochs, hist["val_taxi"],   color="#2ca02c", lw=2, ls="--",
            alpha=0.75, label="Val L_taxi")
    ax.fill_between(epochs, hist["train_taxi"], hist["val_taxi"],
                    alpha=0.08, color="#2ca02c")
    ax.set_title("L_taxi  —  E[Taxi Time] Component\n(minutes, lower = shorter taxi)",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("E[T_taxi] (min)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=8)

    # ── Panel C: L_cong — soft congestion proxy ───────────────────────────
    ax = axes[0, 2]
    ax.plot(epochs, hist["train_cong"], color="#ff7f0e", lw=2, label="Train L_cong")
    ax.plot(epochs, hist["val_cong"],   color="#ff7f0e", lw=2, ls="--",
            alpha=0.75, label="Val L_cong")
    ax.fill_between(epochs, hist["train_cong"], hist["val_cong"],
                    alpha=0.08, color="#ff7f0e")
    ax.set_title("L_cong  —  Soft Congestion Proxy\n(Σ temporal-kernel × same-terminal prob)",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("L_cong", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=8)

    # ── Panel D: Pareto front ──────────────────────────────────────────────
    ax = axes[1, 0]
    if PARE_CSV.exists():
        pareto = pd.read_csv(PARE_CSV)
        sc = ax.scatter(
            pareto["val_taxi"], pareto["val_cong"],
            c=pareto["epoch"], cmap="plasma",
            s=120, edgecolors="black", linewidths=0.6, zorder=3,
        )
        for _, row in pareto.iterrows():
            ax.annotate(
                f"ep{int(row['epoch'])}",
                (row["val_taxi"], row["val_cong"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7,
            )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Epoch", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "pareto_front.csv not found",
                transform=ax.transAxes, ha="center", fontsize=9)
    ax.set_xlabel("Val L_taxi", fontsize=9)
    ax.set_ylabel("Val L_cong", fontsize=9)
    ax.set_title("Pareto-Optimal Val Epochs\n(lower-left is better for both objectives)",
                 fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=8)

    # ── Panel E: Generalisation gap ───────────────────────────────────────
    ax = axes[1, 1]
    gap_taxi = hist["val_taxi"] - hist["train_taxi"]
    gap_cong = hist["val_cong"] - hist["train_cong"]
    bar_w    = 0.4
    ax.bar(epochs - bar_w/2, gap_taxi, width=bar_w,
           color="#2ca02c", alpha=0.75, label="Taxi gap")
    ax.bar(epochs + bar_w/2, gap_cong, width=bar_w,
           color="#ff7f0e", alpha=0.75, label="Cong gap")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Generalisation Gap (Val − Train)\nPositive = val worse than train",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Gap", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, lw=0.5, axis="y")
    ax.tick_params(labelsize=8)

    # ── Panel F: Policy comparison bar chart ──────────────────────────────
    ax = axes[1, 2]
    if COMP_CSV.exists():
        comp     = pd.read_csv(COMP_CSV)
        policies = [p.strip() for p in comp["policy"]]
        taxi_v   = comp["f3_taxi_min"].values
        queue_v  = comp["f3_queue_min"].values
        total_v  = comp["f3_total_min"].values

        x     = np.arange(len(policies))
        bar_w = 0.25
        ax.bar(x - bar_w, taxi_v,  bar_w, color="#2ca02c", alpha=0.8, label="F3 taxi (min)")
        ax.bar(x,          queue_v, bar_w, color="#ff7f0e", alpha=0.8, label="F3 queue (min)")
        bars_total = ax.bar(x + bar_w,  total_v, bar_w, color="#1f77b4", alpha=0.8, label="F3 total (min)")

        for bars, vals in [(bars_total, total_v)]:
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=12, ha="right", fontsize=7)
        ax.set_ylabel("Simulated delay (min)", fontsize=9)
        ax.set_title("Policy Comparison — F3 Queueing Metrics\n(lower total is better)",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, lw=0.5, axis="y")
        ax.tick_params(labelsize=8)
    else:
        ax.text(0.5, 0.5, "policy_comparison.csv not found",
                transform=ax.transAxes, ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
