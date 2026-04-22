"""
Training metrics visualization for ECE538FP GNN.

Reads:
  outputs/training_history.csv   — per-epoch train/val loss, taxi, cong, fill
  outputs/pareto_front.csv       — Pareto-optimal epochs (val_taxi, val_cong, val_fill)
  outputs/policy_comparison.csv  — baseline vs GNN F3 scores

Produces outputs/training_metrics.png with seven panels:
  A. Total loss          — train vs val
  B. L_taxi component    — E[T_taxi] train vs val
  C. L_cong component    — soft congestion proxy train vs val
  D. L_fill component    — gate-zone density penalty train vs val
  E. Pareto front        — val_taxi vs val_cong, sized by val_fill, coloured by epoch
  F. Generalisation gap  — val minus train for taxi, cong, fill
  G. Baseline vs GNN     — F3 queueing metrics bar chart (full-width)

Usage:
    python src/plot_training.py
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
HIST_CSV = ROOT / "outputs" / "training_history.csv"
PARE_CSV = ROOT / "outputs" / "pareto_front.csv"
COMP_CSV = ROOT / "outputs" / "policy_comparison.csv"
OUT_PATH = ROOT / "outputs" / "training_metrics.png"

# Colour palette (consistent across panels)
_C_LOSS  = "#1f77b4"   # blue   — total loss
_C_TAXI  = "#2ca02c"   # green  — taxi
_C_CONG  = "#ff7f0e"   # orange — congestion
_C_FILL  = "#9467bd"   # purple — fill density penalty


def _loss_panel(ax, epochs, train_vals, val_vals, colour, title, ylabel):
    """Shared helper: train/val curve with shaded gap."""
    ax.plot(epochs, train_vals, color=colour, lw=2,       label="Train")
    ax.plot(epochs, val_vals,   color=colour, lw=2, ls="--",
            alpha=0.75, label="Val")
    ax.fill_between(epochs, train_vals, val_vals, alpha=0.08, color=colour)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=8)


def main():
    if not HIST_CSV.exists():
        print(f"ERROR: {HIST_CSV} not found.  Run  python -m src.train  first.")
        return

    hist   = pd.read_csv(HIST_CSV)
    epochs = hist["epoch"].values

    # ── Figure layout: 3 rows × 3 cols, bottom row spans full width ─────────
    fig = plt.figure(figsize=(20, 13), facecolor="#f8f9fa")
    gs  = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.48, wspace=0.32,
        top=0.93, bottom=0.07, left=0.06, right=0.97,
    )

    ax_loss   = fig.add_subplot(gs[0, 0])
    ax_taxi   = fig.add_subplot(gs[0, 1])
    ax_cong   = fig.add_subplot(gs[0, 2])
    ax_fill   = fig.add_subplot(gs[1, 0])
    ax_pareto = fig.add_subplot(gs[1, 1])
    ax_gap    = fig.add_subplot(gs[1, 2])
    ax_comp   = fig.add_subplot(gs[2, :])   # full-width bottom row

    fig.suptitle(
        "GNN Training Metrics — ECE538FP\n"
        "Spatio-Temporal GNN · EWR Gate Scheduling  "
        "(5 zones: A-Wide, A-Narrow, B-Narrow, C-Wide, C-Narrow)",
        fontsize=12, fontweight="bold", color="#1a1a2e",
    )

    # ── Panel A: Total loss ──────────────────────────────────────────────────
    _loss_panel(
        ax_loss, epochs,
        hist["train_loss"], hist["val_loss"],
        _C_LOSS,
        "Total Loss\n(β·L_taxi + λ·L_cong + η·L_fill + γ·L_turn)",
        "Loss",
    )

    # ── Panel B: L_taxi ──────────────────────────────────────────────────────
    _loss_panel(
        ax_taxi, epochs,
        hist["train_taxi"], hist["val_taxi"],
        _C_TAXI,
        "L_taxi  —  E[Taxi Time]  [minutes, same scale as hard sim]",
        "E[T_taxi] (min)",
    )
    # Draw hard-sim baseline reference line if policy_comparison.csv exists
    if COMP_CSV.exists():
        try:
            _comp = pd.read_csv(COMP_CSV)
            _base_row = _comp[_comp["policy"].str.contains("Greedy", case=False)]
            if not _base_row.empty:
                _base_taxi = float(_base_row.iloc[0]["f3_taxi_min"])
                ax_taxi.axhline(
                    _base_taxi, color="red", ls=":", lw=1.4,
                    label=f"Baseline hard-sim ({_base_taxi:.2f} min)",
                )
                ax_taxi.legend(fontsize=7)
        except Exception:
            pass

    # ── Panel C: L_cong ──────────────────────────────────────────────────────
    _loss_panel(
        ax_cong, epochs,
        hist["train_cong"], hist["val_cong"],
        _C_CONG,
        "L_cong  —  Soft Congestion Proxy\n(temporal-kernel × same-zone probability)",
        "L_cong",
    )

    # ── Panel D: L_fill ──────────────────────────────────────────────────────
    has_fill = "train_fill" in hist.columns and "val_fill" in hist.columns
    if has_fill:
        _loss_panel(
            ax_fill, epochs,
            hist["train_fill"], hist["val_fill"],
            _C_FILL,
            "L_fill  —  Gate-Zone Density Penalty\n(quadratic in fill ratio above 30% threshold)",
            "L_fill",
        )
    else:
        ax_fill.text(0.5, 0.5, "train_fill / val_fill\nnot found in history",
                     transform=ax_fill.transAxes, ha="center", va="center",
                     fontsize=9, color="grey")
        ax_fill.set_title("L_fill  —  Gate-Zone Density Penalty",
                          fontweight="bold", fontsize=10)

    # ── Panel E: Pareto front ─────────────────────────────────────────────────
    if PARE_CSV.exists():
        pareto = pd.read_csv(PARE_CSV)

        # Point size inversely proportional to val_fill (lower fill → larger dot)
        if "val_fill" in pareto.columns and pareto["val_fill"].notna().any():
            fill_norm = pareto["val_fill"].values
            fill_norm = (fill_norm - fill_norm.min()) / max(fill_norm.max() - fill_norm.min(), 1e-9)
            sizes = 200 * (1 - fill_norm) + 60   # 60–260 px
            fill_label = "Size ∝ 1/val_fill"
        else:
            sizes = 120
            fill_label = None

        sc = ax_pareto.scatter(
            pareto["val_taxi"], pareto["val_cong"],
            c=pareto["epoch"], cmap="plasma",
            s=sizes, edgecolors="black", linewidths=0.6, zorder=3,
        )
        for _, row in pareto.iterrows():
            ax_pareto.annotate(
                f"ep{int(row['epoch'])}",
                (row["val_taxi"], row["val_cong"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7,
            )
        cbar = fig.colorbar(sc, ax=ax_pareto, pad=0.02)
        cbar.set_label("Epoch", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        if fill_label:
            ax_pareto.text(
                0.03, 0.97, fill_label,
                transform=ax_pareto.transAxes, fontsize=7,
                va="top", ha="left", color="#9467bd",
            )
    else:
        ax_pareto.text(0.5, 0.5, "pareto_front.csv\nnot found",
                       transform=ax_pareto.transAxes, ha="center",
                       va="center", fontsize=9, color="grey")

    ax_pareto.set_xlabel("Val L_taxi", fontsize=9)
    ax_pareto.set_ylabel("Val L_cong", fontsize=9)
    ax_pareto.set_title(
        "Pareto-Optimal Val Epochs\n(lower-left + small point = all 3 objectives better)",
        fontweight="bold", fontsize=10,
    )
    ax_pareto.grid(True, alpha=0.3, lw=0.5)
    ax_pareto.tick_params(labelsize=8)

    # ── Panel F: Generalisation gap ──────────────────────────────────────────
    gap_taxi = (hist["val_taxi"] - hist["train_taxi"]).values
    gap_cong = (hist["val_cong"] - hist["train_cong"]).values
    gap_fill = (hist["val_fill"] - hist["train_fill"]).values if has_fill else np.zeros_like(gap_taxi)

    bar_w = 0.28
    x     = epochs.astype(float)
    ax_gap.bar(x - bar_w,      gap_taxi, width=bar_w, color=_C_TAXI, alpha=0.75, label="Taxi gap")
    ax_gap.bar(x,              gap_cong, width=bar_w, color=_C_CONG, alpha=0.75, label="Cong gap")
    if has_fill:
        ax_gap.bar(x + bar_w, gap_fill, width=bar_w, color=_C_FILL, alpha=0.75, label="Fill gap")
    ax_gap.axhline(0, color="black", lw=0.8)
    ax_gap.set_title("Generalisation Gap  (Val − Train)\nPositive = val worse than train",
                     fontweight="bold", fontsize=10)
    ax_gap.set_xlabel("Epoch", fontsize=9)
    ax_gap.set_ylabel("Gap", fontsize=9)
    ax_gap.legend(fontsize=7)
    ax_gap.grid(True, alpha=0.3, lw=0.5, axis="y")
    ax_gap.tick_params(labelsize=8)

    # ── Panel G: Policy comparison (full-width) ───────────────────────────────
    if COMP_CSV.exists():
        comp     = pd.read_csv(COMP_CSV)
        policies = [p.strip() for p in comp["policy"]]
        taxi_v   = comp["f3_taxi_min"].values
        queue_v  = comp["f3_queue_min"].values
        total_v  = comp["f3_total_min"].values

        n     = len(policies)
        x     = np.arange(n)
        bar_w = 0.25

        b_taxi  = ax_comp.bar(x - bar_w, taxi_v,  bar_w, color=_C_TAXI, alpha=0.85, label="F3 taxi (min)")
        b_queue = ax_comp.bar(x,          queue_v, bar_w, color=_C_CONG, alpha=0.85, label="F3 queue (min)")
        b_total = ax_comp.bar(x + bar_w,  total_v, bar_w, color=_C_LOSS, alpha=0.85, label="F3 total (min)")

        # Label total bars
        for bar, val in zip(b_total, total_v):
            ax_comp.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )
        # Label taxi and queue bars too
        for bar, val in list(zip(b_taxi, taxi_v)) + list(zip(b_queue, queue_v)):
            ax_comp.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#444",
            )

        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels(policies, rotation=10, ha="right", fontsize=9)
        ax_comp.set_ylabel("Simulated delay (min)", fontsize=9)
        ax_comp.set_title(
            "Policy Comparison — F3 Hard-Simulator Queueing Metrics  "
            "(Baseline  |  GNN Greedy  |  GNN + Safe Override)\n"
            "Lower total = better  |  L_taxi (training) is in the same unit as F3 taxi",
            fontweight="bold", fontsize=10,
        )
        ax_comp.legend(fontsize=8, loc="upper right")
        ax_comp.grid(True, alpha=0.3, lw=0.5, axis="y")
        ax_comp.tick_params(labelsize=8)
    else:
        ax_comp.text(
            0.5, 0.5,
            "policy_comparison.csv not found\n"
            "(run  python -m src.train  to generate)",
            transform=ax_comp.transAxes, ha="center", va="center",
            fontsize=10, color="grey",
        )
        ax_comp.set_title("Policy Comparison — F3 Queueing Metrics",
                          fontweight="bold", fontsize=10)

    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
