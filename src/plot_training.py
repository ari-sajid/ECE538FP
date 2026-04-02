"""
Training metrics visualization for ECE538FP GNN.

Reads outputs/training_history.csv and outputs/pareto_front.csv and produces
outputs/training_metrics.png with four panels:

  A. Total loss (train vs test) over epochs
  B. F1 Gate Constraint loss over epochs
  C. F2 Taxiing Distance loss over epochs
  D. F3 Schedule Stability (delay) loss over epochs
  E. Pareto front scatter (F1 vs F2 vs F3, coloured by epoch)
  F. Train vs Test gap (generalisation)

Usage:
    python src/plot_training.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
HIST_CSV = ROOT / "outputs" / "training_history.csv"
PARE_CSV = ROOT / "outputs" / "pareto_front.csv"
OUT_PATH = ROOT / "outputs" / "training_metrics.png"


def main():
    if not HIST_CSV.exists():
        print(f"ERROR: {HIST_CSV} not found. Run training first.")
        return

    hist = pd.read_csv(HIST_CSV)
    epochs = hist["epoch"].values

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#f8f9fa")
    fig.suptitle(
        "GNN Training Metrics — ECE538FP\n"
        "Spatio-Temporal GNN for Airport Gate Scheduling (EWR & LGA)",
        fontsize=13, fontweight="bold", y=1.01, color="#1a1a2e"
    )

    # All three objectives on one panel for direct comparison
    ax_all = axes[0, 0]
    ax_all.plot(epochs, hist["train_F1"], color="#d62728", linewidth=2, label="Train F1")
    ax_all.plot(epochs, hist["test_F1"],  color="#d62728", linewidth=2, linestyle="--", alpha=0.7, label="Test F1")
    ax_all.plot(epochs, hist["train_F2"], color="#2ca02c", linewidth=2, label="Train F2")
    ax_all.plot(epochs, hist["test_F2"],  color="#2ca02c", linewidth=2, linestyle="--", alpha=0.7, label="Test F2")
    ax_all.set_title("F1 & F2 Objectives Over Training\n(F3 on separate scale — see D)", fontweight="bold", fontsize=10)
    ax_all.set_xlabel("Epoch", fontsize=9)
    ax_all.set_ylabel("Loss", fontsize=9)
    ax_all.legend(fontsize=8)
    ax_all.grid(True, alpha=0.3, linewidth=0.5)
    ax_all.tick_params(labelsize=8)

    plot_cfg = [
        ("train_F1", "test_F1", "F1 – Gate Constraint Loss",    "#d62728"),
        ("train_F2", "test_F2", "F2 – Taxiing Distance Loss",   "#2ca02c"),
        ("train_F3", "test_F3", "F3 – Schedule Stability Loss", "#ff7f0e"),
    ]

    for ax, (tr_col, te_col, title, colour) in zip(axes.flat[1:4], plot_cfg):
        ax.plot(epochs, hist[tr_col], color=colour, linewidth=2, label="Train", zorder=3)
        ax.plot(epochs, hist[te_col], color=colour, linewidth=2, linestyle="--",
                alpha=0.7, label="Test", zorder=3)
        ax.fill_between(epochs, hist[tr_col], hist[te_col],
                        alpha=0.08, color=colour)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)

    # Panel E: Pareto front (F1 vs F2, size = F3)
    ax_pareto = axes[1, 1]
    if PARE_CSV.exists():
        pareto = pd.read_csv(PARE_CSV)
        scatter = ax_pareto.scatter(
            pareto["test_F2"], pareto["test_F1"],
            c=pareto["epoch"], cmap="plasma",
            s=(pareto["test_F3"] / pareto["test_F3"].max()) * 400 + 60,
            edgecolors="black", linewidths=0.6, zorder=3, alpha=0.9
        )
        for _, row in pareto.iterrows():
            ax_pareto.annotate(
                f"ep{int(row['epoch'])}",
                (row["test_F2"], row["test_F1"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7
            )
        cbar = fig.colorbar(scatter, ax=ax_pareto, pad=0.02)
        cbar.set_label("Epoch", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax_pareto.set_xlabel("F2 – Taxiing Distance", fontsize=9)
        ax_pareto.set_ylabel("F1 – Gate Constraint", fontsize=9)
        ax_pareto.set_title("Pareto Front (Test Set)\nBubble size ∝ F3 delay penalty",
                            fontweight="bold", fontsize=10)
    else:
        ax_pareto.text(0.5, 0.5, "pareto_front.csv not found",
                       transform=ax_pareto.transAxes, ha="center")
    ax_pareto.grid(True, alpha=0.3, linewidth=0.5)
    ax_pareto.tick_params(labelsize=8)

    # Panel F: Train/Test gap over epochs
    ax_gap = axes[1, 2]
    # Use F3 for the gap since it's the dominant and most volatile objective
    gap = hist["test_F3"] - hist["train_F3"]
    colours = ["#d62728" if g > 0 else "#2ca02c" for g in gap]
    ax_gap.bar(epochs, gap, color=colours, alpha=0.75, width=0.8)
    ax_gap.axhline(0, color="black", linewidth=0.8)
    ax_gap.set_title("Generalisation Gap on F3\n(Test − Train delay loss)",
                     fontweight="bold", fontsize=10)
    ax_gap.set_xlabel("Epoch", fontsize=9)
    ax_gap.set_ylabel("Gap", fontsize=9)
    ax_gap.grid(True, alpha=0.3, linewidth=0.5, axis="y")
    ax_gap.tick_params(labelsize=8)

    from matplotlib.patches import Patch
    ax_gap.legend(handles=[
        Patch(color="#d62728", alpha=0.75, label="Overfitting (test > train)"),
        Patch(color="#2ca02c", alpha=0.75, label="Underfitting (train > test)"),
    ], fontsize=7, loc="upper right")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
