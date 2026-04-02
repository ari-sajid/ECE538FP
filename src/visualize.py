"""
GNN Visualization for Airport Gate Scheduling (ECE538FP).

Produces  outputs/gnn_visualization.png  with six project-specific panels:

  1. EWR Airport Road Network  – OSM graph + terminal & runway locations
  2. LGA Airport Road Network  – same for LaGuardia
  3. Turnaround Edge Subgraph  – 6 aircraft chains on a representative day
  4. Congestion Edge Subgraph  – flights in the busiest 15-min window at EWR
  5. Carrier-Terminal Constraint Heatmap – direct visualisation of gate_mapping.json
  6. Hourly Flight Volume       – departure counts by hour, EWR vs LGA

All data is loaded from the raw / meta / geo files so the script runs even
before the main data-engineering pipeline has been executed.

Usage:
    python src/visualize.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
RAW_CSV  = ROOT / "data" / "raw"  / "nyc_master_2025.csv"
GATE_MAP = ROOT / "data" / "meta" / "gate_mapping.json"
EWR_GML  = ROOT / "data" / "geo"  / "ewr_layout.graphml"
LGA_GML  = ROOT / "data" / "geo"  / "lga_layout.graphml"
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Terminal and runway approximate coordinates
# ---------------------------------------------------------------------------
TERMINAL_COORDS = {
    "EWR": {
        "Terminal A": (40.6892, -74.1751),
        "Terminal B": (40.6927, -74.1739),
        "Terminal C": (40.6963, -74.1734),
    },
    "LGA": {
        "Terminal B": (40.7757, -73.8763),
        "Terminal C": (40.7742, -73.8790),
    },
}
RUNWAY_COORDS = {
    "EWR": (40.6878, -74.1860),
    "LGA": (40.7700, -73.8820),
}

# Fixed colours per carrier so they're consistent across panels
CARRIER_COLOURS = {
    "UA": "#1f77b4",  "AA": "#d62728",  "B6": "#2ca02c",
    "DL": "#9467bd",  "WN": "#ff7f0e",  "NK": "#8c564b",
    "AS": "#e377c2",  "F9": "#bcbd22",  "G4": "#17becf",
    "MQ": "#7f7f7f",  "OO": "#aec7e8",  "YX": "#ffbb78",
}
TERMINAL_COLOURS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


# ===========================================================================
# Panel helpers
# ===========================================================================

def _draw_airport_network(ax, graphml_path: Path, airport: str, zoom_pad: float = 0.012):
    """
    Draw the OSM road network for one airport with terminal + runway overlays.
    Uses LineCollection for speed on the large GraphML graphs.
    """
    print(f"  Loading {airport} GraphML ({graphml_path.stat().st_size // 1024} KB)…")
    G = nx.read_graphml(str(graphml_path))

    # Extract positions dict: node_id → (lon, lat)
    pos = {}
    for node, d in G.nodes(data=True):
        try:
            pos[node] = (float(d["x"]), float(d["y"]))
        except (KeyError, ValueError):
            pass

    if not pos:
        ax.text(0.5, 0.5, "No coordinate data", transform=ax.transAxes, ha="center")
        return

    # Determine viewport centred on terminals + some padding
    t_lats = [c[0] for c in TERMINAL_COORDS[airport].values()]
    t_lons = [c[1] for c in TERMINAL_COORDS[airport].values()]
    r_lat, r_lon = RUNWAY_COORDS[airport]
    all_lats = t_lats + [r_lat]
    all_lons = t_lons + [r_lon]

    lon_min = min(all_lons) - zoom_pad
    lon_max = max(all_lons) + zoom_pad
    lat_min = min(all_lats) - zoom_pad
    lat_max = max(all_lats) + zoom_pad

    # Filter nodes within viewport
    vis_nodes = {n: p for n, p in pos.items()
                 if lon_min <= p[0] <= lon_max and lat_min <= p[1] <= lat_max}

    # Draw edges as LineCollection (fast batch rendering)
    segments = []
    for u, v in G.edges():
        if u in vis_nodes and v in vis_nodes:
            segments.append([vis_nodes[u], vis_nodes[v]])

    if segments:
        lc = LineCollection(segments, linewidths=0.4, colors="#b0b0b0", alpha=0.6, zorder=1)
        ax.add_collection(lc)

    # Draw nodes as tiny dots
    if vis_nodes:
        xs, ys = zip(*vis_nodes.values())
        ax.scatter(xs, ys, s=1.5, c="#999999", alpha=0.4, zorder=2, linewidths=0)

    # Overlay terminals
    for i, (t_name, (t_lat, t_lon)) in enumerate(TERMINAL_COORDS[airport].items()):
        c = TERMINAL_COLOURS[i]
        ax.scatter(t_lon, t_lat, s=220, c=c, marker="*",
                   zorder=6, edgecolors="black", linewidths=0.6)
        ax.annotate(t_name, (t_lon, t_lat),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, fontweight="bold", color=c,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    # Overlay runway
    ax.scatter(r_lon, r_lat, s=120, c="black", marker="D", zorder=6)
    ax.annotate("Runway", (r_lon, r_lat),
                xytext=(4, -10), textcoords="offset points",
                fontsize=7, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title(f"{airport} Airport Network\n(OSM road graph + terminal locations)",
                 fontweight="bold", fontsize=9)
    ax.set_xlabel("Longitude", fontsize=7)
    ax.set_ylabel("Latitude",  fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, linewidth=0.4)

    # Legend for terminals
    handles = [
        mpatches.Patch(color=TERMINAL_COLOURS[i], label=t)
        for i, t in enumerate(TERMINAL_COORDS[airport])
    ]
    handles.append(mpatches.Patch(color="black", label="Runway"))
    ax.legend(handles=handles, fontsize=6.5, loc="best",
              framealpha=0.85, edgecolor="gray")


def _draw_turnaround_subgraph(ax, df: pd.DataFrame):
    """
    Panel 3: directed turnaround chains for 6 representative aircraft on a
    moderately busy mid-year day.
    """
    # Pick a date with decent EWR departures in June
    june = df[df.FL_DATE.str.startswith("2025-06")]
    day = june.groupby("FL_DATE").size().sort_values().iloc[len(june.groupby("FL_DATE")) // 2]
    sample_date = day.name if hasattr(day, "name") else "2025-06-15"

    day_df = df[df.FL_DATE == sample_date].copy()
    day_df = day_df.sort_values(["TAIL_NUM", "CRS_DEP_TIME"])

    # Pick 6 aircraft that each have 3-8 flights that day
    tail_counts = day_df.groupby("TAIL_NUM").size()
    good_tails  = tail_counts[(tail_counts >= 3) & (tail_counts <= 8)].index
    chosen      = list(good_tails[:6])

    if not chosen:
        chosen = tail_counts.nlargest(6).index.tolist()

    G = nx.DiGraph()
    node_meta = {}   # node_id → {tail, carrier, hour, airport}

    for tail in chosen:
        flights = day_df[day_df.TAIL_NUM == tail].reset_index()
        for i, row in flights.iterrows():
            nid = f"{tail}_{i}"
            G.add_node(nid)
            node_meta[nid] = {
                "tail":    tail,
                "carrier": row.OP_UNIQUE_CARRIER,
                "hour":    int(row.CRS_DEP_TIME) // 100,
                "origin":  row.ORIGIN,
            }
        for i in range(len(flights) - 1):
            src = f"{tail}_{i}"
            tgt = f"{tail}_{i+1}"
            G.add_edge(src, tgt)

    # Layout: chain per tail on separate horizontal tracks
    pos = {}
    for t_idx, tail in enumerate(chosen):
        tail_nodes = [n for n in G.nodes if n.startswith(f"{tail}_")]
        tail_nodes = sorted(tail_nodes, key=lambda n: node_meta[n]["hour"])
        for f_idx, nid in enumerate(tail_nodes):
            pos[nid] = (f_idx * 1.6, -t_idx * 1.8)

    node_colours = [
        CARRIER_COLOURS.get(node_meta[n]["carrier"], "#aaaaaa") for n in G.nodes
    ]
    node_labels  = {n: f"{node_meta[n]['origin']}\n{node_meta[n]['hour']:02d}h"
                    for n in G.nodes}

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                           node_size=280, alpha=0.92)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                            font_size=5.5, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#333333",
                           arrows=True, arrowsize=12,
                           connectionstyle="arc3,rad=0.08",
                           width=1.4, alpha=0.8)

    # Tail-number labels on left margin
    for t_idx, tail in enumerate(chosen):
        carrier = day_df[day_df.TAIL_NUM == tail].iloc[0].OP_UNIQUE_CARRIER
        ax.text(-0.7, -t_idx * 1.8, f"{carrier}\n{tail[-4:]}",
                ha="right", va="center", fontsize=6.5,
                color=CARRIER_COLOURS.get(carrier, "#555"), fontweight="bold")

    ax.set_title(f"Turnaround Edge Chains  ({sample_date})\n"
                 f"6 aircraft · each node = one flight · edges = same-tail links",
                 fontweight="bold", fontsize=9)
    ax.axis("off")

    # Carrier legend
    seen_carriers = {node_meta[n]["carrier"] for n in G.nodes}
    handles = [mpatches.Patch(color=CARRIER_COLOURS.get(c, "#aaa"), label=c)
               for c in sorted(seen_carriers)]
    ax.legend(handles=handles, fontsize=6.5, loc="lower right",
              framealpha=0.85, ncol=2, title="Carrier", title_fontsize=7)


def _draw_congestion_subgraph(ax, df: pd.DataFrame):
    """
    Panel 4: flights in the single busiest 15-minute congestion window at EWR.
    """
    ewr = df[df.ORIGIN == "EWR"].copy()
    ewr["dep_min"] = (ewr.CRS_DEP_TIME // 100) * 60 + (ewr.CRS_DEP_TIME % 100)

    # Find the 15-minute window with the most flights (by sliding window)
    best_window_start, best_flights = 0, []
    for t in range(0, 24 * 60 - 15, 5):
        window = ewr[(ewr.dep_min >= t) & (ewr.dep_min < t + 15)]
        if len(window) > len(best_flights):
            best_window_start = t
            best_flights = window

    window_df = best_flights.head(20).reset_index(drop=True)   # cap at 20 for clarity

    G = nx.Graph()
    for i, row in window_df.iterrows():
        G.add_node(i, carrier=row.OP_UNIQUE_CARRIER,
                   dep=row.CRS_DEP_TIME, dist=row.DISTANCE)

    # Add congestion edges (all pairs within the window)
    for i in range(len(window_df)):
        for j in range(i + 1, len(window_df)):
            G.add_edge(i, j)

    pos = nx.circular_layout(G)
    node_colours = [CARRIER_COLOURS.get(G.nodes[n]["carrier"], "#aaa") for n in G.nodes]
    node_labels  = {n: f"{G.nodes[n]['carrier']}\n{G.nodes[n]['dep']:04d}" for n in G.nodes}

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                           node_size=320, alpha=0.92)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                            font_size=5.5, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#e07b39",
                           width=0.8, alpha=0.4)

    h = best_window_start // 60
    m = best_window_start % 60
    ax.set_title(f"Congestion Edge Cluster – EWR\nBusiest 15-min window: "
                 f"{h:02d}:{m:02d}–{h:02d}:{(m+15)%60:02d}  "
                 f"({len(window_df)} flights, all mutually connected)",
                 fontweight="bold", fontsize=9)
    ax.axis("off")

    seen_carriers = {G.nodes[n]["carrier"] for n in G.nodes}
    handles = [mpatches.Patch(color=CARRIER_COLOURS.get(c, "#aaa"), label=c)
               for c in sorted(seen_carriers)]
    ax.legend(handles=handles, fontsize=6.5, loc="lower right",
              framealpha=0.85, ncol=2, title="Carrier", title_fontsize=7)


def _draw_constraint_heatmap(ax, gate_mapping: dict, all_carriers: list):
    """
    Panel 5: binary heatmap of which carriers may use which terminals.
    Directly visualises the F1 gate-constraint loss structure.
    """
    gate_cols = [
        "EWR\nTerm. A", "EWR\nTerm. B", "EWR\nTerm. C",
        "LGA\nTerm. B", "LGA\nTerm. C",
    ]
    gate_keys = [
        ("EWR", "Terminal_A"), ("EWR", "Terminal_B"), ("EWR", "Terminal_C"),
        ("LGA", "Terminal_B"), ("LGA", "Terminal_C"),
    ]

    # Build matrix: rows = carriers, cols = terminals
    matrix = np.zeros((len(all_carriers), len(gate_keys)))
    for col_idx, (airport, terminal) in enumerate(gate_keys):
        allowed = gate_mapping.get(airport, {}).get(terminal, [])
        for row_idx, carrier in enumerate(all_carriers):
            if carrier in allowed:
                matrix[row_idx, col_idx] = 1.0

    # Custom two-tone colour map
    cmap = mcolors.ListedColormap(["#f7f7f7", "#2166ac"])

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(gate_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(all_carriers), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(range(len(gate_cols)))
    ax.set_xticklabels(gate_cols, fontsize=8)
    ax.set_yticks(range(len(all_carriers)))
    ax.set_yticklabels(all_carriers, fontsize=8)

    # Annotate cells
    for r in range(len(all_carriers)):
        for c in range(len(gate_cols)):
            val = "✓" if matrix[r, c] else "✗"
            colour = "white" if matrix[r, c] else "#cccccc"
            ax.text(c, r, val, ha="center", va="center",
                    fontsize=9, color=colour, fontweight="bold")

    ax.set_title("Carrier–Terminal Constraint Matrix\n"
                 "(drives F₁ Gate Constraint Loss)",
                 fontweight="bold", fontsize=9)

    # Colour bar
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2166ac", label="Authorised ✓"),
        Patch(facecolor="#f7f7f7", edgecolor="#aaa", label="Not authorised ✗"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper right",
              framealpha=0.9, bbox_to_anchor=(1.0, -0.08), ncol=2)


def _draw_hourly_volume(ax, df: pd.DataFrame):
    """
    Panel 6: stacked bar chart of hourly departure volume at EWR and LGA,
    coloured by carrier, to show the temporal density driving congestion edges.
    """
    dep_df = df[df.ORIGIN.isin(["EWR", "LGA"])].copy()
    dep_df["hour"] = dep_df.CRS_DEP_TIME // 100

    hours = list(range(5, 24))

    ewr_by_hour = dep_df[dep_df.ORIGIN == "EWR"].groupby("hour").size().reindex(hours, fill_value=0)
    lga_by_hour = dep_df[dep_df.ORIGIN == "LGA"].groupby("hour").size().reindex(hours, fill_value=0)

    # Carrier breakdown for EWR (stacked)
    carriers = sorted(dep_df.OP_UNIQUE_CARRIER.unique())
    ewr_carrier = (
        dep_df[dep_df.ORIGIN == "EWR"]
        .groupby(["hour", "OP_UNIQUE_CARRIER"])
        .size()
        .unstack(fill_value=0)
        .reindex(hours, fill_value=0)
        .reindex(columns=carriers, fill_value=0)
    )
    lga_carrier = (
        dep_df[dep_df.ORIGIN == "LGA"]
        .groupby(["hour", "OP_UNIQUE_CARRIER"])
        .size()
        .unstack(fill_value=0)
        .reindex(hours, fill_value=0)
        .reindex(columns=carriers, fill_value=0)
    )

    x   = np.arange(len(hours))
    bar_w = 0.38

    # EWR stacked bars (left)
    bottom_ewr = np.zeros(len(hours))
    for carrier in carriers:
        vals = ewr_carrier[carrier].values if carrier in ewr_carrier.columns else np.zeros(len(hours))
        ax.bar(x - bar_w / 2, vals, bar_w, bottom=bottom_ewr,
               color=CARRIER_COLOURS.get(carrier, "#aaa"),
               label=f"EWR – {carrier}" if bottom_ewr.sum() == 0 else "_nolegend_",
               alpha=0.88)
        bottom_ewr += vals

    # LGA stacked bars (right, hatched)
    bottom_lga = np.zeros(len(hours))
    for carrier in carriers:
        vals = lga_carrier[carrier].values if carrier in lga_carrier.columns else np.zeros(len(hours))
        ax.bar(x + bar_w / 2, vals, bar_w, bottom=bottom_lga,
               color=CARRIER_COLOURS.get(carrier, "#aaa"),
               hatch="//", alpha=0.65)
        bottom_lga += vals

    # 15-min congestion threshold line: flights_per_hour / 4
    # Shade the congestion zone (top quarter of each bar)
    ax.axhline(y=50, color="red", linewidth=1.0, linestyle="--", alpha=0.6,
               label="≈ 15-min congestion threshold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Annual departures (all 2025)", fontsize=8)
    ax.set_title("Hourly Departure Volume – EWR (solid) vs LGA (hatched)\n"
                 "Bars stacked by carrier · dashed = ~congestion edge density threshold",
                 fontweight="bold", fontsize=9)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Legend: one entry per carrier
    carrier_handles = [
        mpatches.Patch(color=CARRIER_COLOURS.get(c, "#aaa"), label=c)
        for c in carriers
    ]
    airport_handles = [
        mpatches.Patch(facecolor="#888", label="EWR (solid)"),
        mpatches.Patch(facecolor="#888", hatch="//", alpha=0.6, label="LGA (hatched)"),
    ]
    ax.legend(handles=carrier_handles + airport_handles,
              fontsize=6.5, ncol=4, loc="upper left",
              framealpha=0.85, title="Carrier / Airport", title_fontsize=7)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("ECE538FP – GNN Visualization")
    print("=" * 60)

    # ── Load shared data ──────────────────────────────────────────────────
    print("\nLoading flight data…")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df["FL_DATE"] = df["FL_DATE"].astype(str)

    print("Loading gate mapping…")
    with open(GATE_MAP) as fh:
        gate_mapping = json.load(fh)

    all_carriers = sorted(df.OP_UNIQUE_CARRIER.unique())

    # ── Build figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 24), facecolor="#f8f9fa")
    fig.suptitle(
        "Spatio-Temporal GNN for Airport Gate Scheduling  —  ECE538FP\n"
        "EWR & LGA · 512,061 flight nodes · turnaround + congestion edges · 2025 BTS data",
        fontsize=14, fontweight="bold", y=0.98,
        color="#1a1a2e",
    )

    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.28,
                          left=0.06, right=0.97, top=0.94, bottom=0.03)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # Panel labels
    for ax, label in zip([ax1, ax2, ax3, ax4, ax5, ax6],
                         ["A", "B", "C", "D", "E", "F"]):
        ax.text(-0.02, 1.04, f"({label})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="#333", va="bottom")

    # ── Panel A: EWR network ──────────────────────────────────────────────
    print("\nPanel A: EWR airport network…")
    _draw_airport_network(ax1, EWR_GML, "EWR")

    # ── Panel B: LGA network ──────────────────────────────────────────────
    print("Panel B: LGA airport network…")
    _draw_airport_network(ax2, LGA_GML, "LGA")

    # ── Panel C: Turnaround subgraph ──────────────────────────────────────
    print("Panel C: Turnaround subgraph…")
    _draw_turnaround_subgraph(ax3, df)

    # ── Panel D: Congestion subgraph ──────────────────────────────────────
    print("Panel D: Congestion subgraph…")
    _draw_congestion_subgraph(ax4, df)

    # ── Panel E: Carrier-terminal constraint heatmap ──────────────────────
    print("Panel E: Constraint heatmap…")
    _draw_constraint_heatmap(ax5, gate_mapping, all_carriers)

    # ── Panel F: Hourly flight volume ─────────────────────────────────────
    print("Panel F: Hourly volume…")
    _draw_hourly_volume(ax6, df)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "gnn_visualization.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved → {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
