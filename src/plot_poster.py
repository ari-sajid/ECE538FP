"""
Research-poster visualizations for the Airport Gate Scheduling GNN.

Generates a series of publication-quality PNG panels in `outputs/poster/`
that can be dropped directly into a research poster. Each panel is saved
as its own file so it can be sized/arranged independently in the poster.

Panels
------
01_pipeline.png            Three-tier system architecture diagram.
02_graph_schema.png        Heterogeneous graph (turnaround + congestion) schema.
03_terminal_distribution.png  Baseline flight volume by terminal (+ unused EWR-C).
04_delay_distribution.png  DEP_DELAY histogram with F3 threshold marker.
05_hourly_pattern.png      Hourly departure volume stacked by airline + airport.
06_safe_override.png       π* decision breakdown (accepted / F1-reject / KT-reject).
07_metrics_summary.png     F1 / F2 / F3 side-by-side comparison (π₀ vs π*).
08_constraint_matrix.png   Carrier × Terminal authorisation heat-map.

Run
---
    python -m src.plot_poster
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "font.size"         : 11,
    "axes.titlesize"    : 14,
    "axes.labelsize"    : 12,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "figure.dpi"        : 120,
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
})

COLOR_EWR     = "#1f77b4"
COLOR_LGA     = "#ff7f0e"
COLOR_GNN     = "#2ca02c"
COLOR_BASE    = "#7f7f7f"
COLOR_ACCEPT  = "#2ca02c"
COLOR_F1      = "#d62728"
COLOR_KT      = "#ff7f0e"
COLOR_PIPE_1  = "#4C72B0"
COLOR_PIPE_2  = "#55A868"
COLOR_PIPE_3  = "#C44E52"

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs" / "poster"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Panel 01 — Pipeline architecture
# ---------------------------------------------------------------------------
def panel_pipeline():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, color, text_color="white", fs=11):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            linewidth=1.5, edgecolor="black", facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=text_color, fontsize=fs, fontweight="bold")

    def arrow(x1, y1, x2, y2, label="", label_offset=0.25):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->,head_width=0.35,head_length=0.45",
            color="black", linewidth=2,
        ))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + label_offset, label,
                    ha="center", va="bottom", fontsize=10, style="italic")

    # Row layout (all three tiers on same y, 4.2 – 5.6)
    y_tier, h_tier = 4.2, 1.4

    # Input
    box(0.2, 4.5, 1.8, 0.8, "512k\nFlight Records", "#34495E", fs=10)

    # Tier 1 — Greedy baseline
    box(2.6, y_tier, 2.6, h_tier,
        "Tier 1\nGreedy First-Fit\n(π₀)", COLOR_PIPE_1)

    # Tier 2 — GNN
    box(5.7, y_tier, 3.0, h_tier,
        "Tier 2\nSpatio-Temporal GNN\nGATConv × 3", COLOR_PIPE_2)

    # Tier 3 — Safe override
    box(9.2, y_tier, 2.6, h_tier,
        "Tier 3\nSafe Override\n(π*)", COLOR_PIPE_3)

    # Final output
    box(12.2, y_tier, 1.7, h_tier, "Final\nAssignment", "#34495E", fs=10)

    # Safety guard (beneath Tier 3)
    box(8.6, 2.3, 3.8, 0.9,
        "Kendall-Tau guard  +  F1 feasibility",
        "#95A5A6", fs=10)

    # Loss objectives (beneath Tier 2)
    box(5.8, 1.0, 0.9, 0.9,  "F1\nConstraint", "#E74C3C", fs=9)
    box(6.85, 1.0, 0.9, 0.9, "F2\nDistance",   "#E67E22", fs=9)
    box(7.9, 1.0, 0.9, 0.9,  "F3\nStability",  "#F1C40F", text_color="black", fs=9)
    ax.text(7.2, 0.55, "Multi-objective loss  L = α·F1 + β·F2 + γ·F3",
            ha="center", fontsize=10, style="italic")

    # Horizontal flow arrows
    arrow(2.0, 4.9, 2.6, 4.9)                     # input → Tier 1
    arrow(5.2, 4.9, 5.7, 4.9, "π₀ ranking", 0.15) # T1 → T2
    arrow(8.7, 4.9, 9.2, 4.9, "proposals",  0.15) # T2 → T3
    arrow(11.8, 4.9, 12.2, 4.9)                   # T3 → output

    # Data feature lane (input to GNN feature input)
    ax.annotate("",
                xy=(5.7, 4.35), xytext=(1.1, 4.5),
                arrowprops=dict(arrowstyle="->", color="#7F8C8D",
                                linewidth=1.6, linestyle="--",
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(3.5, 3.55, "raw features",
            fontsize=9, style="italic", color="#555555")

    # Guard ← Tier 3 (label the relationship)
    arrow(10.5, 4.2, 10.5, 3.2)                   # T3 queries guard

    # Loss → GNN (training feedback)
    arrow(7.2, 1.9, 7.2, 4.2)

    ax.text(7.0, 6.4,
            "Three-tier Airport Gate Scheduling Pipeline",
            ha="center", fontsize=16, fontweight="bold")

    fig.savefig(OUTDIR / "01_pipeline.png")
    plt.close(fig)
    print("  [01] pipeline.png")


# ---------------------------------------------------------------------------
# Panel 02 — Heterogeneous graph schema
# ---------------------------------------------------------------------------
def panel_graph_schema():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Turnaround edges (same aircraft, consecutive flights): top row
    y_turn = 4.5
    turn_x = [1.0, 2.8, 4.6, 6.4, 8.2]
    for i, x in enumerate(turn_x):
        circ = plt.Circle((x, y_turn), 0.32, color=COLOR_EWR,
                          ec="black", zorder=3)
        ax.add_patch(circ)
        ax.text(x, y_turn, f"f{i+1}", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold", zorder=4)
        if i < 4:
            ax.annotate("", xy=(turn_x[i+1]-0.35, y_turn),
                        xytext=(x+0.35, y_turn),
                        arrowprops=dict(arrowstyle="->", color=COLOR_EWR,
                                        linewidth=2.2))
    ax.text(0.2, y_turn + 0.7, "Turnaround edges (same aircraft → delay propagation)",
            fontsize=11, fontweight="bold", color=COLOR_EWR)

    # Congestion edges (same airport, 15-min window): bottom row
    # Place nodes in two staggered rows so the mesh of edges is visible
    y_cong_top = 2.3
    y_cong_bot = 1.3
    cong_pts = [
        (1.5, y_cong_top), (2.8, y_cong_bot), (4.1, y_cong_top),
        (5.4, y_cong_bot), (6.7, y_cong_top), (8.0, y_cong_bot),
        (9.3, y_cong_top),
    ]
    # Draw edges first (low zorder) so nodes sit on top
    pairs = [(0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4),
             (3,5),(4,5),(4,6),(5,6)]
    for i, j in pairs:
        xi, yi = cong_pts[i]
        xj, yj = cong_pts[j]
        ax.plot([xi, xj], [yi, yj],
                color=COLOR_LGA, linewidth=2.0, alpha=0.55, zorder=1)
    for k, (x, y) in enumerate(cong_pts):
        circ = plt.Circle((x, y), 0.32, color=COLOR_LGA,
                          ec="black", zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, f"g{k+1}", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold", zorder=4)
    y_cong = 1.8
    ax.text(0.2, y_cong - 0.9,
            "Congestion edges (same airport, ±15 min window → runway competition)",
            fontsize=11, fontweight="bold", color=COLOR_LGA)

    # Title
    ax.text(5.0, 5.7, "Heterogeneous Flight Graph",
            ha="center", fontsize=15, fontweight="bold")

    # Stats box
    stats_text = ("Nodes      : 512,061 flights\n"
                  "Edge type 1: turnaround (aircraft continuity)\n"
                  "Edge type 2: congestion (15-min proximity)\n"
                  "Features   : 141-dim per node")
    ax.text(9.9, 3.2, stats_text, ha="right", va="center", fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECEFF4",
                      edgecolor="gray"))

    fig.savefig(OUTDIR / "02_graph_schema.png")
    plt.close(fig)
    print("  [02] graph_schema.png")


# ---------------------------------------------------------------------------
# Panel 03 — Terminal distribution
# ---------------------------------------------------------------------------
def panel_terminal_distribution(baseline_path):
    df = pd.read_csv(baseline_path)
    counts = df["gate_name"].value_counts().reindex(
        ["EWR_Terminal_A", "EWR_Terminal_B", "EWR_Terminal_C",
         "LGA_Terminal_B", "LGA_Terminal_C"]).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ["EWR\nTerminal A", "EWR\nTerminal B", "EWR\nTerminal C",
              "LGA\nTerminal B", "LGA\nTerminal C"]
    colors = [COLOR_EWR, COLOR_EWR, "#9FC8E8", COLOR_LGA, "#FFB580"]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="black")

    # Annotations
    for bar, count in zip(bars, counts.values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 4000,
                f"{count:,}", ha="center", fontsize=11, fontweight="bold")

    # Highlight unused EWR-C
    if counts["EWR_Terminal_C"] == 0:
        ax.annotate("Unused by baseline\n→ GNN opportunity",
                    xy=(2, 2000), xytext=(2, 90000),
                    ha="center", fontsize=11, color=COLOR_F1,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=COLOR_F1, lw=2))

    ax.set_ylabel("Flights Assigned (2025)")
    ax.set_title("Baseline (π₀) Terminal Utilisation",
                 fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(counts.values) * 1.18)

    fig.savefig(OUTDIR / "03_terminal_distribution.png")
    plt.close(fig)
    print("  [03] terminal_distribution.png")


# ---------------------------------------------------------------------------
# Panel 04 — Delay distribution
# ---------------------------------------------------------------------------
def panel_delay_distribution(flights_path):
    df = pd.read_csv(flights_path, usecols=["DEP_DELAY"])
    delays = df["DEP_DELAY"].dropna()
    clipped = delays.clip(-30, 120)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(clipped, bins=75, color=COLOR_EWR, edgecolor="black", alpha=0.85)

    # F3 (loss definition) = mean of ReLU(delay) — same stat baseline.py reports
    f3_relu     = delays.clip(lower=0).mean()
    delay_rate  = (delays > 15).mean() * 100
    median      = delays.median()

    ax.axvline(0, color="black", linestyle="--", linewidth=1.4, label="On-time")
    ax.axvline(f3_relu, color=COLOR_F1, linewidth=2.2,
               label=f"F3 = mean ReLU(delay) = {f3_relu:.2f} min")

    ax.set_xlabel("Departure Delay (minutes)")
    ax.set_ylabel("Flight Count")
    ax.set_title("Delay Distribution — 2025 NYC Departures",
                 fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    # Stats inset
    txt = (f"n = {len(delays):,}\n"
           f"Delayed (>15 min): {delay_rate:.1f}%\n"
           f"Median delay    : {median:.1f} min\n"
           f"F3 (ReLU mean)  : {f3_relu:.2f} min")
    ax.text(0.98, 0.62, txt, transform=ax.transAxes, ha="right", va="top",
            family="monospace", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECEFF4",
                      edgecolor="gray"))

    fig.savefig(OUTDIR / "04_delay_distribution.png")
    plt.close(fig)
    print("  [04] delay_distribution.png")


# ---------------------------------------------------------------------------
# Panel 05 — Hourly pattern
# ---------------------------------------------------------------------------
def panel_hourly_pattern(flights_path):
    df = pd.read_csv(flights_path,
                     usecols=["FL_DATE", "ORIGIN", "CRS_DEP_TIME"])
    df = df[df["ORIGIN"].isin(["EWR", "LGA"])].copy()
    df["hour"] = (df["CRS_DEP_TIME"] // 100).astype(int)
    n_days = df["FL_DATE"].nunique()

    fig, ax = plt.subplots(figsize=(11, 6))
    hours = np.arange(5, 24)
    for origin, color, hatch in [("EWR", COLOR_EWR, ""),
                                  ("LGA", COLOR_LGA, "//")]:
        sub = df[df["ORIGIN"] == origin]
        annual = sub.groupby("hour").size().reindex(hours, fill_value=0)
        per_day = annual.values / max(n_days, 1)      # avg flights / hour / day
        offset = -0.2 if origin == "EWR" else 0.2
        ax.bar(hours + offset, per_day, width=0.4,
               label=origin, color=color, hatch=hatch, edgecolor="black",
               alpha=0.9)

    # Congestion threshold (per airport, per hour, per day)
    ax.axhline(30, color=COLOR_F1, linestyle="--", linewidth=1.8,
               label="Congestion threshold (30 flights/h)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Departures per Hour (per day)")
    ax.set_title(f"Hourly Departure Volume — EWR vs LGA  "
                 f"(averaged over {n_days} days, 2025)",
                 fontweight="bold", pad=12)
    ax.set_xticks(hours)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(OUTDIR / "05_hourly_pattern.png")
    plt.close(fig)
    print("  [05] hourly_pattern.png")


# ---------------------------------------------------------------------------
# Panel 06 — Safe-override decision breakdown
# ---------------------------------------------------------------------------
def panel_safe_override(decisions_path):
    df = pd.read_csv(decisions_path)
    categories = ["accepted", "rejected_f1_violation", "rejected_kt_exceeded"]
    labels     = ["Accepted\n(GNN proposal)", "Rejected\n(F1 violation)",
                  "Rejected\n(KT guard)"]
    colors     = [COLOR_ACCEPT, COLOR_F1, COLOR_KT]

    counts = [int((df["decision"] == c).sum()) for c in categories]
    total  = sum(counts)
    pcts   = [c / total * 100 for c in counts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                   gridspec_kw={"width_ratios":[1, 1.3]})

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        counts, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, wedgeprops=dict(edgecolor="black", linewidth=1.2),
        textprops=dict(fontsize=11),
    )
    for a in autotexts:
        a.set_color("white")
        a.set_fontweight("bold")
    ax1.set_title(f"π* Decision Breakdown  (n = {total:,})",
                  fontweight="bold", pad=12)

    # Horizontal bar chart (raw counts)
    y_pos = np.arange(len(labels))[::-1]
    ax2.barh(y_pos, counts, color=colors, edgecolor="black")
    for y, c, p in zip(y_pos, counts, pcts):
        ax2.text(c + total * 0.01, y, f"{c:,}  ({p:.1f}%)",
                 va="center", fontsize=11)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Flights")
    ax2.set_xlim(0, max(counts) * 1.25)
    ax2.set_title("Decision Volume  (test split)", fontweight="bold", pad=12)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Safe-Override Policy (π*) — Safety Guard Activation",
                 fontweight="bold", fontsize=15, y=1.01)

    fig.savefig(OUTDIR / "06_safe_override.png")
    plt.close(fig)
    print("  [06] safe_override.png")


# ---------------------------------------------------------------------------
# Panel 07 — Metrics summary
# ---------------------------------------------------------------------------
def panel_metrics_summary(comparison_path):
    df = pd.read_csv(comparison_path)
    policies = df["policy"].tolist()
    f1s = df["f1_violations"].tolist()
    f2s = df["f2_mean_dist_m"].tolist()
    f3s = df["f3_mean_pos_delay_min"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    bar_colors = [COLOR_BASE, COLOR_GNN]

    for ax, values, title, unit in zip(
        axes,
        [f1s, f2s, f3s],
        ["F1: Gate-Constraint Violations",
         "F2: Mean Taxi Distance",
         "F3: Mean Positive Delay"],
        ["violations", "metres", "minutes"],
    ):
        bars = ax.bar(policies, values, color=bar_colors,
                      edgecolor="black", width=0.55)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(values, default=1)*0.02 + 0.1,
                    f"{v:,.2f}" if isinstance(v, float) else f"{v:,}",
                    ha="center", fontsize=11, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.3)
        ymax = max(values) if max(values) > 0 else 1
        ax.set_ylim(0, ymax * 1.25)

    fig.suptitle("Multi-Objective Performance  (π₀  vs.  π*)",
                 fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / "07_metrics_summary.png")
    plt.close(fig)
    print("  [07] metrics_summary.png")


# ---------------------------------------------------------------------------
# Panel 08 — Carrier × Terminal constraint matrix
# ---------------------------------------------------------------------------
def panel_constraint_matrix(gate_map_path):
    with open(gate_map_path) as fh:
        mapping = json.load(fh)

    terminals = ["EWR_Terminal_A", "EWR_Terminal_B", "EWR_Terminal_C",
                 "LGA_Terminal_B", "LGA_Terminal_C"]
    # Flatten: {airport: {terminal: [carriers]}} → unique carrier set
    carriers = sorted({
        c
        for airport_map in mapping.values()
        for carrier_list in airport_map.values()
        for c in carrier_list
    })
    # Ensure common majors appear even if absent from the mapping
    for must in ["UA", "AA", "DL", "WN", "B6", "AS", "9E", "NK", "F9"]:
        if must not in carriers:
            carriers.append(must)
    carriers = sorted(set(carriers))

    # Build binary matrix carrier × terminal
    M = np.zeros((len(carriers), len(terminals)), dtype=int)
    for t_idx, t_name in enumerate(terminals):
        airport, suffix = t_name.split("_", 1)          # "EWR", "Terminal_A"
        term_carriers = mapping.get(airport, {}).get(suffix, [])
        for c_idx, c in enumerate(carriers):
            if c in term_carriers:
                M[c_idx, t_idx] = 1

    fig, ax = plt.subplots(figsize=(8, max(5, len(carriers)*0.25)))
    cmap = plt.cm.Blues
    im = ax.imshow(M, cmap=cmap, aspect="auto",
                   interpolation="nearest", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(terminals)))
    ax.set_xticklabels([t.replace("_Terminal_", "-") for t in terminals],
                       rotation=30, ha="right")
    ax.set_yticks(np.arange(len(carriers)))
    ax.set_yticklabels(carriers)

    # Gridlines
    ax.set_xticks(np.arange(-0.5, len(terminals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(carriers), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate cells
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j]:
                ax.text(j, i, "✓", ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")

    ax.set_title("Carrier × Terminal Authorisation Matrix (F1 feasibility)",
                 fontweight="bold", pad=12)

    legend_elems = [
        mpatches.Patch(facecolor=cmap(0.85), edgecolor="black",
                       label="Authorised"),
        mpatches.Patch(facecolor=cmap(0.0),  edgecolor="black",
                       label="Prohibited (F1 violation)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right",
              bbox_to_anchor=(1.45, 1.0), framealpha=0.95)

    fig.savefig(OUTDIR / "08_constraint_matrix.png")
    plt.close(fig)
    print("  [08] constraint_matrix.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Saving poster panels to: {OUTDIR}\n")

    baseline_csv    = ROOT / "outputs" / "baseline_assignments.csv"
    comparison_csv  = ROOT / "outputs" / "policy_comparison.csv"
    decisions_csv   = ROOT / "outputs" / "safe_override_decisions.csv"
    flights_csv     = ROOT / "data"    / "raw" / "nyc_master_2025.csv"
    gate_map        = ROOT / "data"    / "meta" / "gate_mapping.json"

    # Diagrammatic (no data required)
    panel_pipeline()
    panel_graph_schema()

    # Data-backed
    if baseline_csv.exists():
        panel_terminal_distribution(baseline_csv)
    else:
        print(f"  [skip] {baseline_csv.name} not found")

    if flights_csv.exists():
        panel_delay_distribution(flights_csv)
        panel_hourly_pattern(flights_csv)
    else:
        print(f"  [skip] {flights_csv.name} not found")

    if decisions_csv.exists():
        panel_safe_override(decisions_csv)
    else:
        print(f"  [skip] {decisions_csv.name} not found")

    if comparison_csv.exists():
        panel_metrics_summary(comparison_csv)
    else:
        print(f"  [skip] {comparison_csv.name} not found")

    if gate_map.exists():
        panel_constraint_matrix(gate_map)
    else:
        print(f"  [skip] {gate_map.name} not found")

    print(f"\nDone. Files written to {OUTDIR}")


if __name__ == "__main__":
    main()
