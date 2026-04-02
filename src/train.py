"""
Training script for the Spatio-Temporal GNN airport gate scheduler.

Run from the project root:
    python -m src.train [--epochs 50] [--batch_size 512] ...

Data split
----------
* Months 01-10 → training set
* Months 11-12 → test set  (unseen future flights)

Memory efficiency
-----------------
The raw graph has ~512 k nodes and potentially millions of edges.
PyG's NeighborLoader is used so that only small subgraphs (seed nodes +
sampled 2-hop neighbourhoods) are materialised per mini-batch, keeping
peak RAM well below the full-graph size.

Temporal batching (Step 5)
--------------------------
Training seed nodes are sorted into (month, time-window) buckets before
being fed to NeighborLoader.  Each mini-batch therefore contains flights
from the same departure window so the model sees complete intra-day
congestion clusters together rather than random cross-day mixes.
Use --window_hours to control bucket width (default 4 h).

Attention heads (Step 6)
------------------------
The model uses GATConv; --num_heads controls how many attention heads are
used per layer (default 4).  hidden_channels must be divisible by num_heads.

Pareto front
------------
After training, the three test-set objective values (F1, F2, F3) recorded
at each epoch are analysed for dominance.  The non-dominated (Pareto-optimal)
epochs are written to  outputs/pareto_front.csv.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

# Allow running as   python src/train.py   (adds project root to path)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import SpatioTemporalGNN, NUM_GATES  # noqa: E402
from src.loss import MultiObjectiveLoss              # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

NODE_CSV = DATA_DIR / "processed" / "final_node_features.csv"
EDGE_CSV = DATA_DIR / "processed" / "edges.csv"
RAW_CSV  = DATA_DIR / "raw"       / "nyc_master_2025.csv"
GATE_MAP = DATA_DIR / "meta"      / "gate_mapping.json"
EWR_GML  = DATA_DIR / "geo"       / "ewr_layout.graphml"
LGA_GML  = DATA_DIR / "geo"       / "lga_layout.graphml"

# ---------------------------------------------------------------------------
# Default hyper-parameters (all overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    hidden_channels  = 128,
    num_layers       = 3,
    num_heads        = 4,    # GAT attention heads (Step 6)
    dropout          = 0.3,
    lr               = 1e-3,
    weight_decay     = 1e-4,
    epochs           = 50,
    batch_size       = 512,
    num_neighbors_l1 = 10,   # neighbours sampled at hop 1
    num_neighbors_l2 = 5,    # neighbours sampled at hop 2
    window_hours     = 4,    # temporal batch window width in hours (Step 5)
    alpha            = 1.0,  # F1 weight
    beta             = 1.0,  # F2 weight
    gamma            = 1.0,  # F3 weight
    lam              = 1.0,  # auxiliary delay-regression weight
    device           = "cuda" if torch.cuda.is_available() else "cpu",
)


# ---------------------------------------------------------------------------
# Temporal batch ordering (Step 5)
# ---------------------------------------------------------------------------

def temporal_sort_ids(
    node_ids: torch.Tensor,
    crs_dep_times: np.ndarray,
    fl_months: np.ndarray,
    window_hours: int = 4,
) -> torch.Tensor:
    """
    Return node_ids sorted by (month, departure-time window) so that each
    NeighborLoader mini-batch contains flights from the same time window.

    Parameters
    ----------
    node_ids      : 1-D LongTensor of global node indices
    crs_dep_times : ndarray[int] of shape (N_total,) – CRS_DEP_TIME (HHMM)
    fl_months     : ndarray[int] of shape (N_total,) – FL_DATE month (1-12)
    window_hours  : width of each time bucket in hours (default 4)

    Returns
    -------
    LongTensor of the same length as node_ids, sorted temporally.
    """
    ids_np   = node_ids.numpy()
    hours    = (crs_dep_times[ids_np] // 100).astype(np.int32)
    windows  = (hours // window_hours).astype(np.int32)
    months   = fl_months[ids_np].astype(np.int32)
    # Composite sort key: month takes precedence, then intra-day window
    sort_key = months * 1000 + windows
    order    = np.argsort(sort_key, kind="stable")
    return torch.from_numpy(ids_np[order])


# ---------------------------------------------------------------------------
# Pareto-front utilities
# ---------------------------------------------------------------------------

def _dominates(a, b):
    """True if point a weakly dominates b on all objectives and strictly on one."""
    return (all(ai <= bi for ai, bi in zip(a, b))
            and any(ai < bi for ai, bi in zip(a, b)))


def pareto_front_indices(points):
    """Return indices of non-dominated points (lower is better)."""
    front = []
    for i, p in enumerate(points):
        if not any(_dominates(points[j], p) for j in range(len(points)) if j != i):
            front.append(i)
    return front


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _check_files():
    missing = [p for p in (NODE_CSV, EDGE_CSV, RAW_CSV, GATE_MAP, EWR_GML, LGA_GML)
               if not p.exists()]
    if missing:
        print("\nMissing required files:")
        for p in missing:
            print(f"  {p}")
        if NODE_CSV in missing or EDGE_CSV in missing:
            print("\nRun the data-engineering pipeline first:")
            print("  python src/clean_weather.py")
            print("  python src/finalize_data.py")
            print("  python src/graph_engine.py")
        raise FileNotFoundError("Required data files are missing (see above).")


def load_node_features(node_csv: Path, raw_csv: Path):
    """
    Load final_node_features.csv and return tensors needed for training.

    Returns
    -------
    x            : FloatTensor [N, F]   – node feature matrix
    delay        : FloatTensor [N]      – observed departure delay (minutes)
    carrier_ohe  : FloatTensor [N, C]   – one-hot carrier vectors
    is_lga       : FloatTensor [N]      – 1 if the flight uses LGA, 0 for EWR
    is_at_nyc    : FloatTensor [N]      – 1 if flight touches EWR or LGA
    train_mask   : BoolTensor  [N]      – months 1-10
    test_mask    : BoolTensor  [N]      – months 11-12
    carrier_list : list[str]            – carrier codes in OHE column order
    """
    print(f"Loading node features ({node_csv.name})…")
    df = pd.read_csv(node_csv, low_memory=False)

    # Train / test split on FL_DATE month
    fl_date = pd.to_datetime(df["FL_DATE"])
    train_mask = (fl_date.dt.month <= 10).values
    test_mask  = (fl_date.dt.month  > 10).values
    print(f"  Train months 1-10 : {train_mask.sum():,} flights")
    print(f"  Test  months 11-12: {test_mask.sum():,} flights")

    # Identify one-hot column groups
    carrier_cols = sorted(c for c in df.columns if c.startswith("OP_UNIQUE_CARRIER_"))
    origin_cols  = sorted(c for c in df.columns if c.startswith("ORIGIN_"))
    carrier_list = [c.replace("OP_UNIQUE_CARRIER_", "") for c in carrier_cols]
    origin_list  = [c.replace("ORIGIN_", "") for c in origin_cols]

    # Departure delay: supervised regression target, clipped to [-60, 300] min
    delay_raw = pd.to_numeric(df["DEP_DELAY"], errors="coerce").fillna(0.0)
    delay = delay_raw.clip(-60, 300).values.astype(np.float32)

    # ── Airport assignment (Step 3) ─────────────────────────────────────────
    # AT_EWR and AT_LGA are now pre-computed by finalize_data.py and stored
    # directly in the node feature matrix — no raw CSV reload needed.
    if "AT_EWR" in df.columns and "AT_LGA" in df.columns:
        is_ewr    = df["AT_EWR"].astype(np.float32).values
        is_lga    = df["AT_LGA"].astype(np.float32).values
        is_at_nyc = np.clip(is_ewr + is_lga, 0, 1).astype(np.float32)
        print("  Airport flags     : AT_EWR / AT_LGA loaded from node features.")
    else:
        # Fallback for node feature files generated before Step 3
        print("  AT_EWR/AT_LGA not found — falling back to ORIGIN OHE columns.")
        is_ewr = df.get("ORIGIN_EWR", pd.Series(0, index=df.index)).astype(float).values
        is_lga = df.get("ORIGIN_LGA", pd.Series(0, index=df.index)).astype(float).values
        is_at_nyc = np.clip(is_ewr + is_lga, 0, 1).astype(np.float32)

    # ── Feature matrix ──────────────────────────────────────────────────────
    # Drop non-numeric / metadata columns; DEP_DELAY used separately above.
    skip = {"FL_DATE", "DEP_TIME", "DEP_DELAY"}
    feature_cols = [c for c in df.columns if c not in skip]

    # Convert everything to numeric; fill NaNs with column mean, then 0
    feat_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    col_means = feat_df.mean()
    feat_df = feat_df.fillna(col_means).fillna(0.0)
    x = feat_df.values.astype(np.float32)                       # [N, F]

    carrier_ohe = df[carrier_cols].astype(np.float32).values    # [N, C]

    # CRS_DEP_TIME and FL_DATE month — used by temporal batch sorter
    crs_dep_times = pd.to_numeric(
        df.get("CRS_DEP_TIME", pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0).values.astype(np.int32)
    fl_months = fl_date.dt.month.values.astype(np.int32)

    print(f"  Feature dimension : {x.shape[1]}")
    print(f"  Carriers tracked  : {len(carrier_list)} ({', '.join(carrier_list[:8])}…)")

    return (
        torch.from_numpy(x),
        torch.from_numpy(delay),
        torch.from_numpy(carrier_ohe),
        torch.from_numpy(is_lga),
        torch.from_numpy(is_at_nyc),
        torch.from_numpy(train_mask),
        torch.from_numpy(test_mask),
        carrier_list,
        origin_list,
        feature_cols,
        crs_dep_times,
        fl_months,
    )


def load_edges(edge_csv: Path):
    """
    Load edges.csv and split into turnaround / congestion edge_index tensors.

    Returns
    -------
    turnaround_ei : LongTensor [2, E_t]
    congestion_ei : LongTensor [2, E_c]
    """
    print(f"Loading edges ({edge_csv.name})…")
    edges = pd.read_csv(edge_csv, low_memory=False)

    ta = edges[edges["type"] == "turnaround"][["source", "target"]].values.T
    cg = edges[edges["type"] == "congestion"][["source", "target"]].values.T

    turnaround_ei = torch.from_numpy(ta.astype(np.int64))
    congestion_ei = torch.from_numpy(cg.astype(np.int64))

    print(f"  Turnaround edges  : {turnaround_ei.shape[1]:,}")
    print(f"  Congestion edges  : {congestion_ei.shape[1]:,}")
    return turnaround_ei, congestion_ei


def build_hetero_data(
    x: torch.Tensor,
    delay: torch.Tensor,
    turnaround_ei: torch.Tensor,
    congestion_ei: torch.Tensor,
) -> HeteroData:
    """Assemble a PyG HeteroData object from the processed arrays."""
    data = HeteroData()

    data["flight"].x         = x        # [N, F]
    data["flight"].y_delay   = delay    # [N]
    data["flight"].num_nodes = x.size(0)

    data["flight", "turnaround", "flight"].edge_index = turnaround_ei
    data["flight", "congestion", "flight"].edge_index  = congestion_ei

    return data


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device,
                carrier_ohe_all, is_lga_all, is_at_nyc_all):
    model.train()
    totals = {"loss": 0.0, "f1": 0.0, "f2": 0.0, "f3": 0.0}
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        # Global indices of seed nodes in this mini-batch
        seed_ids = batch["flight"].input_id   # [B]
        n_seed   = seed_ids.size(0)

        gate_logits, delay_pred = model(
            {"flight": batch["flight"].x},
            batch.edge_index_dict,
        )

        # Restrict outputs and targets to seed nodes only
        gate_logits_s = gate_logits[:n_seed]
        delay_pred_s  = delay_pred[:n_seed]
        delay_true_s  = batch["flight"].y_delay[:n_seed].to(device)

        # Per-flight metadata (loaded from CPU, indexed by global seed IDs)
        c_ohe = carrier_ohe_all[seed_ids].to(device)
        lga   = is_lga_all[seed_ids].to(device)
        nyc   = is_at_nyc_all[seed_ids].to(device)

        loss, f1, f2, f3 = criterion(
            gate_logits_s, delay_pred_s, delay_true_s,
            c_ohe, lga, nyc,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals["loss"] += loss.item()
        totals["f1"]   += f1.item()
        totals["f2"]   += f2.item()
        totals["f3"]   += f3.item()
        n_batches      += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# One evaluation epoch (no gradients)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_epoch(model, loader, criterion, device,
               carrier_ohe_all, is_lga_all, is_at_nyc_all):
    model.eval()
    totals = {"loss": 0.0, "f1": 0.0, "f2": 0.0, "f3": 0.0}
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        seed_ids = batch["flight"].input_id
        n_seed   = seed_ids.size(0)

        gate_logits, delay_pred = model(
            {"flight": batch["flight"].x},
            batch.edge_index_dict,
        )

        gate_logits_s = gate_logits[:n_seed]
        delay_pred_s  = delay_pred[:n_seed]
        delay_true_s  = batch["flight"].y_delay[:n_seed].to(device)

        c_ohe = carrier_ohe_all[seed_ids].to(device)
        lga   = is_lga_all[seed_ids].to(device)
        nyc   = is_at_nyc_all[seed_ids].to(device)

        loss, f1, f2, f3 = criterion(
            gate_logits_s, delay_pred_s, delay_true_s,
            c_ohe, lga, nyc,
        )

        totals["loss"] += loss.item()
        totals["f1"]   += f1.item()
        totals["f2"]   += f2.item()
        totals["f3"]   += f3.item()
        n_batches      += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train the Spatio-Temporal GNN for airport gate scheduling"
    )
    for key, val in DEFAULTS.items():
        if isinstance(val, bool):
            p.add_argument(f"--{key}", type=lambda x: x.lower() == "true",
                           default=val)
        else:
            p.add_argument(f"--{key}", type=type(val), default=val)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device(args.device)
    print(f"Device : {device}\n")

    # ── Verify all data files exist ──────────────────────────────────────
    _check_files()

    # ── Load processed data ──────────────────────────────────────────────
    (x, delay, carrier_ohe, is_lga, is_at_nyc,
     train_mask, test_mask, carrier_list, origin_list,
     feature_cols, crs_dep_times, fl_months) = load_node_features(NODE_CSV, RAW_CSV)

    turnaround_ei, congestion_ei = load_edges(EDGE_CSV)
    data = build_hetero_data(x, delay, turnaround_ei, congestion_ei)

    N     = x.size(0)
    F_in  = x.size(1)
    train_ids = train_mask.nonzero(as_tuple=False).squeeze(1)
    test_ids  = test_mask.nonzero(as_tuple=False).squeeze(1)
    print(f"\nGraph  : {N:,} nodes | F_in={F_in}")
    print(f"Train  : {len(train_ids):,} seed nodes")
    print(f"Test   : {len(test_ids):,} seed nodes\n")

    # ── Temporal sort for training (Step 5) ──────────────────────────────
    # Sort train seed nodes by (month, departure-time window) so each
    # mini-batch processes flights from the same congestion window together.
    print(f"Sorting train nodes into {args.window_hours}-hour temporal windows…")
    train_ids_sorted = temporal_sort_ids(
        train_ids, crs_dep_times, fl_months, args.window_hours
    )
    print(f"  Done.  First batch window: "
          f"month={fl_months[train_ids_sorted[0].item()]} "
          f"dep={crs_dep_times[train_ids_sorted[0].item()]:04d}\n")

    # ── Build NeighborLoaders ─────────────────────────────────────────────
    num_nbrs = [args.num_neighbors_l1, args.num_neighbors_l2]
    edge_types = [
        ("flight", "turnaround", "flight"),
        ("flight", "congestion",  "flight"),
    ]
    shared_loader_kwargs = dict(
        data           = data,
        num_neighbors  = {et: num_nbrs for et in edge_types},
        batch_size     = args.batch_size,
        num_workers    = 2,
        persistent_workers = True,
    )
    # Training: temporally ordered (shuffle=False preserves window grouping)
    train_loader = NeighborLoader(
        input_nodes=("flight", train_ids_sorted),
        shuffle=False,
        **shared_loader_kwargs,
    )
    # Test: original order is fine (evaluation only)
    test_loader = NeighborLoader(
        input_nodes=("flight", test_ids),
        shuffle=False,
        **shared_loader_kwargs,
    )

    # ── Model (Step 6: GATConv with num_heads) ────────────────────────────
    model = SpatioTemporalGNN(
        in_channels     = F_in,
        hidden_channels = args.hidden_channels,
        num_gates       = NUM_GATES,
        num_layers      = args.num_layers,
        dropout         = args.dropout,
        num_heads       = args.num_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model  : {n_params:,} trainable parameters\n")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── Loss (builds distance table at init – one-time cost) ─────────────
    criterion = MultiObjectiveLoss(
        gate_mapping_path = str(GATE_MAP),
        ewr_graphml       = str(EWR_GML),
        lga_graphml       = str(LGA_GML),
        carrier_list      = carrier_list,
        alpha             = args.alpha,
        beta              = args.beta,
        gamma             = args.gamma,
        lam               = args.lam,
    ).to(device)

    # ── Training loop ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Epoch':>6}  {'Tr-Loss':>9}  {'Tr-F1':>8}  {'Tr-F2':>8}  {'Tr-F3':>8}"
          f"  {'Te-Loss':>9}  {'Te-F1':>8}  {'Te-F2':>8}  {'Te-F3':>8}")
    print("=" * 80)

    history = []          # [(epoch, tr_f1, tr_f2, tr_f3, te_f1, te_f2, te_f3)]
    best_test_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, device,
                         carrier_ohe, is_lga, is_at_nyc)
        te = eval_epoch( model, test_loader,  criterion, device,
                         carrier_ohe, is_lga, is_at_nyc)
        scheduler.step()

        row = (epoch,
               tr["f1"], tr["f2"], tr["f3"],
               te["f1"], te["f2"], te["f3"])
        history.append(row)

        print(f"{epoch:6d}  {tr['loss']:9.4f}  {tr['f1']:8.4f}  {tr['f2']:8.4f}  "
              f"{tr['f3']:8.4f}  {te['loss']:9.4f}  {te['f1']:8.4f}  {te['f2']:8.4f}  "
              f"{te['f3']:8.4f}")

        # Checkpoint best model by total test loss
        if te["loss"] < best_test_loss:
            best_test_loss = te["loss"]
            torch.save(
                {
                    "epoch"        : epoch,
                    "model_state"  : model.state_dict(),
                    "in_channels"  : F_in,
                    "carrier_list" : carrier_list,
                    "origin_list"  : origin_list,
                    "feature_cols" : feature_cols,
                    "args"         : vars(args),
                },
                OUT_DIR / "best_model.pt",
            )
            # Also save a human-readable feature schema for the inference API
            import json as _json
            schema = {
                "in_channels"  : F_in,
                "carrier_list" : carrier_list,
                "origin_list"  : origin_list,
                "feature_cols" : feature_cols,
                "num_heads"    : args.num_heads,
                "hidden_channels": args.hidden_channels,
                "num_layers"   : args.num_layers,
            }
            with open(OUT_DIR / "feature_schema.json", "w") as _fh:
                _json.dump(schema, _fh, indent=2)

    # ── Pareto front analysis (on test objectives) ────────────────────────
    test_pts = [(r[4], r[5], r[6]) for r in history]   # (F1, F2, F3) at test
    front_idx = pareto_front_indices(test_pts)

    pareto_rows = [
        {
            "epoch":   history[i][0],
            "test_F1": history[i][4],
            "test_F2": history[i][5],
            "test_F3": history[i][6],
        }
        for i in front_idx
    ]
    pareto_df = pd.DataFrame(pareto_rows).sort_values("test_F1")
    pareto_df.to_csv(OUT_DIR / "pareto_front.csv", index=False)

    print("\n" + "=" * 80)
    print("Pareto-optimal epochs on the test set (non-dominated trade-offs):")
    print(pareto_df.to_string(index=False))

    # Save full epoch history
    hist_df = pd.DataFrame(
        history,
        columns=["epoch", "train_F1", "train_F2", "train_F3",
                 "test_F1",  "test_F2",  "test_F3"],
    )
    hist_df.to_csv(OUT_DIR / "training_history.csv", index=False)
    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
