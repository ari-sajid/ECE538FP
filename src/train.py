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
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
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

# Max taxi distance in metres (= APPROX_DIST_M["EWR_Terminal_C"]).
# Multiply normalised F2 by this to get actual metres for reporting.
F2_MAX_DIST_M = 1380.0

# ---------------------------------------------------------------------------
# Default hyper-parameters (all overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    hidden_channels  = 128,
    num_layers       = 3,
    num_heads        = 4,    # GAT attention heads
    dropout          = 0.2,  # reduced from 0.3 — was too aggressive
    lr               = 5e-4, # reduced from 1e-3 for more stable convergence
    weight_decay     = 1e-4,
    epochs           = 100,
    batch_size       = 512,
    num_neighbors_l1 = 10,
    num_neighbors_l2 = 5,
    num_neighbors_l3 = 5,   # must match num_layers=3
    window_hours     = 4,
    # Loss weights — F1 is enforced by hard masking (always 0); alpha is symbolic
    alpha            = 5.0,  # F1 weight (infeasible logits masked to -inf first)
    beta             = 1.0,  # F2 (taxi distance)
    gamma            = 1.0,  # F3 (schedule stability, now normalised)
    lam              = 0.1,  # L_reg (delay regression) — reduced from 1.0
    delay_scale      = 30.0, # divide delay by this to normalise F3 / L_reg
    entropy_weight   = 0.0,  # entropy penalty (0 = off; sign fixed so >0 encourages peaked predictions)
    delta            = 0.0,  # F4 load-balancing weight (disabled: variance minimization conflicts with F2)
    patience         = 15,   # early-stopping patience (val epochs without improvement)
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
    missing = [p for p in (NODE_CSV, EDGE_CSV, RAW_CSV, GATE_MAP)
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

    Split: months 1-8 → train, months 9-10 → val, months 11-12 → test.
    Features are standardised (zero mean, unit variance) using statistics
    computed on the training split only (no leakage into val/test).

    Returns
    -------
    x            : FloatTensor [N, F]   – standardised node feature matrix
    delay        : FloatTensor [N]      – observed departure delay (minutes)
    carrier_ohe  : FloatTensor [N, C]   – one-hot carrier vectors (unscaled)
    is_lga       : FloatTensor [N]      – 1 if the flight uses LGA, 0 for EWR
    is_at_nyc    : FloatTensor [N]      – 1 if flight touches EWR or LGA
    train_mask   : BoolTensor  [N]      – months 1-8
    val_mask     : BoolTensor  [N]      – months 9-10
    test_mask    : BoolTensor  [N]      – months 11-12
    carrier_list : list[str]            – carrier codes in OHE column order
    """
    print(f"Loading node features ({node_csv.name})…")
    df = pd.read_csv(node_csv, low_memory=False)

    # Three-way temporal split (no leakage)
    fl_date    = pd.to_datetime(df["FL_DATE"])
    months     = fl_date.dt.month.values
    train_mask = (months <= 8)
    val_mask   = (months >= 9) & (months <= 10)
    test_mask  = (months > 10)
    print(f"  Train months 1-8  : {train_mask.sum():,} flights")
    print(f"  Val   months 9-10 : {val_mask.sum():,} flights")
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
    skip = {"FL_DATE", "DEP_TIME", "DEP_DELAY"}
    feature_cols = [c for c in df.columns if c not in skip]

    feat_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    # Replace ±inf before computing means — fillna only touches NaN, not inf,
    # so any inf value would make col_means=inf → scaler produces NaN everywhere.
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    col_means = feat_df.mean()
    feat_df = feat_df.fillna(col_means).fillna(0.0)
    x_raw = feat_df.values.astype(np.float32)                   # [N, F]

    # ── Feature standardisation (fit on train only, apply to all) ───────────
    # StandardScaler brings high-range columns (DISTANCE ~2000, HHMM ~1200)
    # into the same scale as the OHE binary columns, giving the input
    # projection stable initial gradients.
    scaler = StandardScaler()
    scaler.fit(x_raw[train_mask])
    x = scaler.transform(x_raw).astype(np.float32)
    # Safety net: clip any residual NaN/inf that survived standardisation.
    x = np.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
    print(f"  Features standardised on {train_mask.sum():,} training rows.")

    carrier_ohe = df[carrier_cols].astype(np.float32).values    # [N, C]

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
        torch.from_numpy(val_mask),
        torch.from_numpy(test_mask),
        carrier_list,
        origin_list,
        feature_cols,
        crs_dep_times,
        fl_months,
    )


def load_edges(edge_csv: Path):
    """
    Load edges.csv and return edge_index tensors for all three edge types.

    Returns
    -------
    turnaround_ei    : LongTensor [2, E_t]
    congestion_ei    : LongTensor [2, E_c]
    same_terminal_ei : LongTensor [2, E_s]  (empty tensor if not in CSV)
    """
    print(f"Loading edges ({edge_csv.name})…")
    edges = pd.read_csv(edge_csv, low_memory=False)

    ta = edges[edges["type"] == "turnaround"][["source", "target"]].values.T
    cg = edges[edges["type"] == "congestion"][["source", "target"]].values.T
    st_rows = edges[edges["type"] == "same_terminal"][["source", "target"]]
    st = st_rows.values.T if len(st_rows) else np.zeros((2, 0), dtype=np.int64)

    turnaround_ei    = torch.from_numpy(ta.astype(np.int64))
    congestion_ei    = torch.from_numpy(cg.astype(np.int64))
    same_terminal_ei = torch.from_numpy(st.astype(np.int64))

    print(f"  Turnaround edges    : {turnaround_ei.shape[1]:,}")
    print(f"  Congestion edges    : {congestion_ei.shape[1]:,}")
    print(f"  Same-terminal edges : {same_terminal_ei.shape[1]:,}")
    return turnaround_ei, congestion_ei, same_terminal_ei


def build_hetero_data(
    x: torch.Tensor,
    delay: torch.Tensor,
    turnaround_ei: torch.Tensor,
    congestion_ei: torch.Tensor,
    same_terminal_ei: torch.Tensor,
    crs_dep_times: np.ndarray,
) -> HeteroData:
    """Assemble a PyG HeteroData object from the processed arrays."""
    data = HeteroData()

    data["flight"].x          = x
    data["flight"].y_delay    = delay
    data["flight"].num_nodes  = x.size(0)
    data["flight"].dep_bucket = torch.from_numpy((crs_dep_times // 30).astype(np.int64))

    data["flight", "turnaround",    "flight"].edge_index = turnaround_ei
    data["flight", "congestion",    "flight"].edge_index = congestion_ei
    data["flight", "same_terminal", "flight"].edge_index = same_terminal_ei

    return data


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device,
                carrier_ohe_all, is_lga_all, is_at_nyc_all):
    model.train()
    totals = {"loss": 0.0, "f2": 0.0, "f3": 0.0, "f3_raw": 0.0}
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        n_seed   = batch["flight"].input_id.size(0)
        # n_id gives global graph node IDs for all subgraph nodes;
        # first n_seed entries are the seed nodes' global indices.
        global_ids = batch["flight"].n_id[:n_seed]

        gate_logits, delay_pred = model(
            {"flight": batch["flight"].x},
            batch.edge_index_dict,
        )

        gate_logits_s = gate_logits[:n_seed]
        delay_pred_s  = delay_pred[:n_seed]
        delay_true_s  = batch["flight"].y_delay[:n_seed].to(device)
        dep_bucket    = batch["flight"].dep_bucket[:n_seed].to(device)

        c_ohe = carrier_ohe_all[global_ids.cpu()].to(device)
        lga   = is_lga_all[global_ids.cpu()].to(device)
        nyc   = is_at_nyc_all[global_ids.cpu()].to(device)

        loss, f2, f3, f3_raw = criterion(
            gate_logits_s, delay_pred_s, delay_true_s,
            c_ohe, lga, nyc, dep_bucket,
        )

        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals["loss"]   += loss.item()
        totals["f2"]     += f2.item()
        totals["f3"]     += f3.item()
        totals["f3_raw"] += f3_raw.item()
        n_batches        += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# One evaluation epoch (no gradients)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_epoch(model, loader, criterion, device,
               carrier_ohe_all, is_lga_all, is_at_nyc_all):
    model.eval()
    totals = {"loss": 0.0, "f2": 0.0, "f3": 0.0, "f3_raw": 0.0}
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        n_seed     = batch["flight"].input_id.size(0)
        global_ids = batch["flight"].n_id[:n_seed]

        gate_logits, delay_pred = model(
            {"flight": batch["flight"].x},
            batch.edge_index_dict,
        )

        gate_logits_s = gate_logits[:n_seed]
        delay_pred_s  = delay_pred[:n_seed]
        delay_true_s  = batch["flight"].y_delay[:n_seed].to(device)
        dep_bucket    = batch["flight"].dep_bucket[:n_seed].to(device)

        c_ohe = carrier_ohe_all[global_ids.cpu()].to(device)
        lga   = is_lga_all[global_ids.cpu()].to(device)
        nyc   = is_at_nyc_all[global_ids.cpu()].to(device)

        loss, f2, f3, f3_raw = criterion(
            gate_logits_s, delay_pred_s, delay_true_s,
            c_ohe, lga, nyc, dep_bucket,
        )

        if not torch.isfinite(loss):
            continue

        totals["loss"]   += loss.item()
        totals["f2"]     += f2.item()
        totals["f3"]     += f3.item()
        totals["f3_raw"] += f3_raw.item()
        n_batches        += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Inference pass — collect gate probabilities for all seed nodes
# ---------------------------------------------------------------------------

@torch.no_grad()
def inference_pass(model, loader, criterion, device, carrier_ohe_all, is_lga_all):
    """
    Run the trained model over a loader and return gate probabilities
    (after GateMasker) and the corresponding local input indices.

    Returns
    -------
    gate_probs_np : ndarray[float32, (N, NUM_GATES)]
    local_ids_np  : ndarray[int64,   (N,)]  — index within loader's input_nodes set
    """
    model.eval()
    all_probs, all_ids = [], []

    for batch in loader:
        batch      = batch.to(device)
        local_ids  = batch["flight"].input_id          # position within input_nodes
        n_seed     = local_ids.size(0)
        global_ids = batch["flight"].n_id[:n_seed]     # global graph node IDs

        gate_logits, _ = model(
            {"flight": batch["flight"].x},
            batch.edge_index_dict,
        )
        gate_logits_s = gate_logits[:n_seed]
        c_ohe = carrier_ohe_all[global_ids.cpu()].to(device)
        lga   = is_lga_all[global_ids.cpu()].to(device)

        masked = criterion.masker.mask_logits(gate_logits_s, c_ohe, lga)
        probs  = torch.softmax(masked, dim=-1)

        all_probs.append(probs.cpu().numpy())
        all_ids.append(local_ids.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_ids)


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
     train_mask, val_mask, test_mask, carrier_list, origin_list,
     feature_cols, crs_dep_times, fl_months) = load_node_features(NODE_CSV, RAW_CSV)

    turnaround_ei, congestion_ei, same_terminal_ei = load_edges(EDGE_CSV)
    data = build_hetero_data(x, delay, turnaround_ei, congestion_ei, same_terminal_ei,
                             crs_dep_times)

    N     = x.size(0)
    F_in  = x.size(1)
    train_ids = train_mask.nonzero(as_tuple=False).squeeze(1)
    val_ids   = val_mask.nonzero(as_tuple=False).squeeze(1)
    test_ids  = test_mask.nonzero(as_tuple=False).squeeze(1)
    print(f"\nGraph  : {N:,} nodes | F_in={F_in}")
    print(f"Train  : {len(train_ids):,}  Val: {len(val_ids):,}  Test: {len(test_ids):,}\n")

    # ── Temporal sort for training ────────────────────────────────────────
    print(f"Sorting train nodes into {args.window_hours}-hour temporal windows…")
    train_ids_sorted = temporal_sort_ids(
        train_ids, crs_dep_times, fl_months, args.window_hours
    )
    print(f"  Done.  First window: "
          f"month={fl_months[train_ids_sorted[0].item()]} "
          f"dep={crs_dep_times[train_ids_sorted[0].item()]:04d}\n")

    # ── Build NeighborLoaders ─────────────────────────────────────────────
    num_nbrs = [args.num_neighbors_l1, args.num_neighbors_l2, args.num_neighbors_l3]
    edge_types = [
        ("flight", "turnaround",    "flight"),
        ("flight", "congestion",    "flight"),
        ("flight", "same_terminal", "flight"),
    ]
    # Windows spawns fresh processes per worker, each loading the full CUDA stack
    # into the page file.  With persistent_workers=True, all three loaders keep
    # their workers alive simultaneously (6 processes), which exhausts the page
    # file.  Use num_workers=0 on Windows so no subprocesses are spawned.
    _n_workers  = 0 if platform.system() == "Windows" else 2
    shared_loader_kwargs = dict(
        data               = data,
        num_neighbors      = {et: num_nbrs for et in edge_types},
        batch_size         = args.batch_size,
        num_workers        = _n_workers,
        persistent_workers = _n_workers > 0,
    )
    train_loader = NeighborLoader(
        input_nodes=("flight", train_ids_sorted), shuffle=False,
        **shared_loader_kwargs,
    )
    val_loader = NeighborLoader(
        input_nodes=("flight", val_ids), shuffle=False,
        **shared_loader_kwargs,
    )
    test_loader = NeighborLoader(
        input_nodes=("flight", test_ids), shuffle=False,
        **shared_loader_kwargs,
    )

    # ── Model ─────────────────────────────────────────────────────────────
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
    # Warm-up for first 5 epochs then cosine annealing
    warmup     = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=5
    )
    cosine     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - 5, 1), eta_min=1e-6
    )
    scheduler  = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[5]
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = MultiObjectiveLoss(
        gate_mapping_path = str(GATE_MAP),
        carrier_list      = carrier_list,
        alpha             = args.alpha,
        beta              = args.beta,
        gamma             = args.gamma,
        lam               = args.lam,
        delay_scale       = args.delay_scale,
        entropy_weight    = args.entropy_weight,
        delta             = args.delta,
    ).to(device)

    # ── Save epoch-0 checkpoint so best_model.pt always exists ─────────────
    import json as _json
    _schema = {
        "in_channels"    : F_in,
        "carrier_list"   : carrier_list,
        "origin_list"    : origin_list,
        "feature_cols"   : feature_cols,
        "num_heads"      : args.num_heads,
        "hidden_channels": args.hidden_channels,
        "num_layers"     : args.num_layers,
    }
    torch.save(
        {
            "epoch"        : 0,
            "model_state"  : model.state_dict(),
            "in_channels"  : F_in,
            "carrier_list" : carrier_list,
            "origin_list"  : origin_list,
            "feature_cols" : feature_cols,
            "args"         : vars(args),
        },
        OUT_DIR / "best_model.pt",
    )
    with open(OUT_DIR / "feature_schema.json", "w") as _fh:
        _json.dump(_schema, _fh, indent=2)

    # ── Training loop with early stopping on val loss ─────────────────────
    print("\n" + "=" * 84)
    print("Constrained GNN Scheduler  |  gate masking enforced structurally (not as loss)")
    print(f"{'Ep':>4}  {'TrLoss':>8}  {'Tr-F2':>7}  {'Tr-F3':>7}"
          f"  {'VaLoss':>8}  {'Va-F2':>7}  {'Va-F3':>7}  {'*':>2}")
    print("=" * 84)

    history         = []
    best_val_loss   = float("inf")
    no_improve      = 0

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, device,
                         carrier_ohe, is_lga, is_at_nyc)
        va = eval_epoch( model, val_loader,   criterion, device,
                         carrier_ohe, is_lga, is_at_nyc)
        scheduler.step()

        improved = va["loss"] < best_val_loss
        marker   = "*" if improved else ""
        if improved:
            best_val_loss = va["loss"]
            no_improve    = 0
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
            schema = {
                "in_channels"    : F_in,
                "carrier_list"   : carrier_list,
                "origin_list"    : origin_list,
                "feature_cols"   : feature_cols,
                "num_heads"      : args.num_heads,
                "hidden_channels": args.hidden_channels,
                "num_layers"     : args.num_layers,
            }
            with open(OUT_DIR / "feature_schema.json", "w") as _fh:
                _json.dump(schema, _fh, indent=2)
        else:
            no_improve += 1

        # history: (epoch, tr_f2, tr_f3, tr_f3_raw, va_f2, va_f3, va_f3_raw)
        row = (epoch,
               tr["f2"], tr["f3"], tr["f3_raw"],
               va["f2"], va["f3"], va["f3_raw"])
        history.append(row)

        print(f"{epoch:4d}  {tr['loss']:8.4f}  {tr['f2']:7.4f}  {tr['f3']:7.4f}"
              f"  {va['loss']:8.4f}  {va['f2']:7.4f}  {va['f3']:7.4f}  {marker}")

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no val improvement for {args.patience} epochs).")
            break

    # ── Final test-set evaluation using best checkpoint ───────────────────
    best_ckpt = torch.load(OUT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    te = eval_epoch(model, test_loader, criterion, device,
                    carrier_ohe, is_lga, is_at_nyc)
    print(f"\nTest  (best ckpt epoch {best_ckpt['epoch']}):"
          f"  F2={te['f2']:.4f} ({te['f2'] * F2_MAX_DIST_M:.0f} m)"
          f"  F3={te['f3']:.4f}  mean-pos-delay={te['f3_raw']:.1f} min")

    # ── Pareto front — 2 objectives: F2 (taxi) and F3 (delay risk) ────────
    val_pts   = [(r[4], r[5]) for r in history]   # (va_f2, va_f3)
    front_idx = pareto_front_indices(val_pts)

    pareto_rows = [
        {"epoch": history[i][0],
         "val_F2": history[i][4], "val_F3": history[i][5]}
        for i in front_idx
    ]
    pareto_df = pd.DataFrame(pareto_rows).sort_values("val_F2")
    pareto_df.to_csv(OUT_DIR / "pareto_front.csv", index=False)
    print("\nPareto-optimal epochs (val set, lower F2 and F3 is better):")
    print(pareto_df.to_string(index=False))

    # Save full epoch history
    hist_df = pd.DataFrame(
        history,
        columns=["epoch",
                 "train_F2", "train_F3", "train_F3_raw",
                 "val_F2",   "val_F3",   "val_F3_raw"],
    )
    hist_df.to_csv(OUT_DIR / "training_history.csv", index=False)

    # Save test metrics to a separate file for clean reporting
    test_row = pd.DataFrame([{
        "best_epoch"         : best_ckpt["epoch"],
        "test_F2"            : te["f2"],
        "test_F3_delay_risk" : te["f3"],
        "test_F3_raw_min"    : te["f3_raw"],
    }])
    test_row.to_csv(OUT_DIR / "test_metrics.csv", index=False)

    # ── Safe Override policy comparison on test set ───────────────────────
    from src.safe_override import SafeOverridePolicy   # noqa: E402
    from src.baseline import GreedyFirstFit            # noqa: E402
    from src.baselines import GreedyBalanced           # noqa: E402

    print("\nRunning Safe Override policy comparison on test set...")

    # Load raw CSV for carrier / airport / dep_time columns
    raw_df = pd.read_csv(RAW_CSV, low_memory=False)
    raw_df["FL_DATE"] = pd.to_datetime(raw_df["FL_DATE"])
    test_df = raw_df[raw_df["FL_DATE"].dt.month > 10].reset_index(drop=True)

    # Baseline assignments + metrics
    base_policy  = GreedyFirstFit(str(GATE_MAP))
    base_assigns = base_policy.assign(test_df)
    base_metrics = base_policy.score(base_assigns, test_df)

    # GreedyBalanced — more realistic heuristic baseline
    bal_policy   = GreedyBalanced(str(GATE_MAP))
    bal_assigns  = bal_policy.assign(test_df)
    bal_metrics  = bal_policy.score(bal_assigns, test_df)

    # Real GNN gate probabilities via inference
    gate_probs_np, seed_ids_np = inference_pass(
        model, test_loader, criterion, device, carrier_ohe, is_lga
    )

    # local_ids are positions within test_ids (= row index in test_df, since
    # test_ids[r] is the global node for test_df row r).
    ordered_probs = np.full((len(test_df), NUM_GATES), 1.0 / NUM_GATES, dtype=np.float32)
    for i, local_id in enumerate(seed_ids_np):
        ordered_probs[int(local_id)] = gate_probs_np[i]

    carriers  = test_df["OP_UNIQUE_CARRIER"].values
    airports  = np.where(test_df["ORIGIN"].isin(["EWR", "LGA"]),
                         test_df["ORIGIN"].values, None)
    # Encode date + time-of-day as minutes since epoch so capacity buckets
    # are per-date (not cumulative across all test days at the same clock time).
    # FL_DATE is datetime64[us]; divide by 10^6*60 to get minutes since epoch.
    fl_dates_min = (pd.to_datetime(test_df["FL_DATE"]).astype(np.int64) // (10**6 * 60)).values
    dep_min_of_day = ((test_df["CRS_DEP_TIME"].values // 100) * 60
                      + (test_df["CRS_DEP_TIME"].values % 100))
    dep_times = (fl_dates_min + dep_min_of_day).astype(np.int64)

    # Apply safe override with real model gate probabilities
    override_policy = SafeOverridePolicy()
    final_assigns, decision_log, override_stats = override_policy.apply(
        ordered_probs, base_assigns, carriers, airports, dep_times
    )
    override_metrics = base_policy.score(final_assigns, test_df)

    def _pct(new, old):
        return f"{(new - old) / max(abs(old), 1) * 100:>+7.1f}%"

    # Print comparison table (FirstFit | Balanced | GNN pi*)
    print(f"\n  {'Metric':<28} {'FirstFit':>10}  {'Balanced':>10}  {'GNN pi*':>10}")
    print("  " + "-" * 65)
    bd  = base_metrics["f2_mean_dist_m"]
    bld = bal_metrics["f2_mean_dist_m"]
    od  = override_metrics["f2_mean_dist_m"]
    print(f"  {'F2 mean taxi dist (m)':<28} {bd:>10.1f}  {bld:>10.1f}  {od:>10.1f}  "
          f"(GNN vs Balanced: {_pct(od, bld)})")
    bf  = base_metrics["f3_mean_pos_delay_min"]
    blf = bal_metrics["f3_mean_pos_delay_min"]
    of  = override_metrics["f3_mean_pos_delay_min"]
    print(f"  {'F3 mean pos delay (min)':<28} {bf:>10.2f}  {blf:>10.2f}  {of:>10.2f}  "
          f"(GNN vs Balanced: {_pct(of, blf)})")

    n = override_stats["n_total"]
    pct_stat = lambda k: f"{100 * override_stats[k] / max(n, 1):.1f}%"
    print(f"\n  Accepted by GNN          : {override_stats['n_accepted_gnn']:,} ({pct_stat('n_accepted_gnn')})")
    print(f"  Low confidence (baseline): {override_stats['n_low_confidence']:,} ({pct_stat('n_low_confidence')})")
    print(f"  Capacity repaired        : {override_stats['n_capacity_repaired']:,}")
    print(f"  Mean gate confidence     : {override_stats['mean_confidence']:.3f}")
    print(f"  Conf threshold           : {override_stats['conf_threshold']}")

    # Save outputs
    decision_log.to_csv(OUT_DIR / "safe_override_decisions.csv", index=False)
    pd.DataFrame([
        {"policy": "GreedyFirstFit (pi0)",
         "f2_mean_dist_m": base_metrics["f2_mean_dist_m"],
         "f3_mean_pos_delay_min": base_metrics["f3_mean_pos_delay_min"]},
        {"policy": "GreedyBalanced (alpha=0.5)",
         "f2_mean_dist_m": bal_metrics["f2_mean_dist_m"],
         "f3_mean_pos_delay_min": bal_metrics["f3_mean_pos_delay_min"]},
        {"policy": "GNN Safe Override",
         "f2_mean_dist_m": override_metrics["f2_mean_dist_m"],
         "f3_mean_pos_delay_min": override_metrics["f3_mean_pos_delay_min"]},
    ]).to_csv(OUT_DIR / "policy_comparison.csv", index=False)

    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
