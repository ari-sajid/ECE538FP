"""
Grid search over loss-weight configurations for the airport gate GNN.

Systematically explores the (α, β, γ, λ) Pareto trade-off space and saves
a ranked summary so you can pick the best configuration before a final run.

Each config trains a full model and reports:
  - Best val F1 / F2 / F3
  - Final test F1 / F2 / F3 (from best checkpoint)
  - Total training time

Usage
-----
    python -m src.grid_search [--epochs 50] [--device cuda]

Results are written to:
    outputs/grid_search_results.csv   – all configs ranked by val composite score
    outputs/grid_search/<config_id>/  – per-config checkpoints and history

Configurations
--------------
Six configurations spanning different Pareto regions:

  Config  alpha  beta  gamma  lam   entropy_weight  Focus
  ──────  ─────  ────  ─────  ────  ──────────────  ──────────────────────────
  A       3.0    1.0   0.5    0.05  0.15  Strongly emphasise gate feasibility (F1)
  B       2.0    2.0   0.5    0.05  0.10  Balanced F1+F2, reduce delay noise
  C       2.0    1.0   1.5    0.05  0.10  Emphasise schedule stability (F3)
  D       1.5    0.5   0.5    0.01  0.20  High-entropy push, loose constraints
  E       4.0    0.5   0.5    0.05  0.10  Aggressive F1 dominance
  F       2.0    1.0   1.0    0.10  0.10  Balanced (matches train.py defaults)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "grid_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Search grid
# ---------------------------------------------------------------------------
CONFIGS = [
    dict(config_id="A", alpha=3.0, beta=1.0, gamma=0.5, lam=0.05, entropy_weight=0.15,
         label="Strong-F1"),
    dict(config_id="B", alpha=2.0, beta=2.0, gamma=0.5, lam=0.05, entropy_weight=0.10,
         label="Balanced-F1-F2"),
    dict(config_id="C", alpha=2.0, beta=1.0, gamma=1.5, lam=0.05, entropy_weight=0.10,
         label="Emphasise-F3"),
    dict(config_id="D", alpha=1.5, beta=0.5, gamma=0.5, lam=0.01, entropy_weight=0.20,
         label="High-entropy"),
    dict(config_id="E", alpha=4.0, beta=0.5, gamma=0.5, lam=0.05, entropy_weight=0.10,
         label="Aggressive-F1"),
    dict(config_id="F", alpha=2.0, beta=1.0, gamma=1.0, lam=0.10, entropy_weight=0.10,
         label="Default-balanced"),
]


def run_config(cfg: dict, base_args: argparse.Namespace) -> dict:
    """Run one training configuration as a subprocess and parse its output."""
    cfg_out = OUT_DIR / cfg["config_id"]
    cfg_out.mkdir(exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.train",
        "--epochs",          str(base_args.epochs),
        "--batch_size",      str(base_args.batch_size),
        "--device",          base_args.device,
        "--patience",        str(base_args.patience),
        "--alpha",           str(cfg["alpha"]),
        "--beta",            str(cfg["beta"]),
        "--gamma",           str(cfg["gamma"]),
        "--lam",             str(cfg["lam"]),
        "--entropy_weight",  str(cfg["entropy_weight"]),
        "--delay_scale",     "30.0",
    ]

    print(f"\n{'='*70}")
    print(f"Config {cfg['config_id']} — {cfg['label']}")
    print(f"  α={cfg['alpha']}  β={cfg['beta']}  γ={cfg['gamma']}"
          f"  λ={cfg['lam']}  ε={cfg['entropy_weight']}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=False, text=True, cwd=str(ROOT)
    )
    elapsed = time.time() - t0

    # Copy outputs into per-config directory
    for fname in ("best_model.pt", "training_history.csv",
                  "test_metrics.csv", "pareto_front.csv"):
        src = ROOT / "outputs" / fname
        if src.exists():
            import shutil
            shutil.copy2(src, cfg_out / fname)

    # Read metrics
    row = {
        "config_id"     : cfg["config_id"],
        "label"         : cfg["label"],
        "alpha"         : cfg["alpha"],
        "beta"          : cfg["beta"],
        "gamma"         : cfg["gamma"],
        "lam"           : cfg["lam"],
        "entropy_weight": cfg["entropy_weight"],
        "elapsed_s"     : round(elapsed),
        "returncode"    : result.returncode,
    }

    test_csv = cfg_out / "test_metrics.csv"
    if test_csv.exists():
        tm = pd.read_csv(test_csv).iloc[0]
        row.update({
            "best_epoch": int(tm["best_epoch"]),
            "test_F1"   : round(float(tm["test_F1"]), 5),
            "test_F2"   : round(float(tm["test_F2"]), 5),
            "test_F3"   : round(float(tm["test_F3"]), 3),
        })
    else:
        row.update({"best_epoch": None, "test_F1": None,
                    "test_F2": None, "test_F3": None})

    # Best val metrics from history
    hist_csv = cfg_out / "training_history.csv"
    if hist_csv.exists():
        hist = pd.read_csv(hist_csv)
        row["best_val_F1"] = round(hist["val_F1"].min(), 5)
        row["best_val_F2"] = round(hist["val_F2"].min(), 5)
        row["best_val_F3"] = round(hist["val_F3"].min(), 3)
        row["epochs_run"]  = len(hist)
    else:
        row.update({"best_val_F1": None, "best_val_F2": None,
                    "best_val_F3": None, "epochs_run": None})

    return row


def composite_score(row: dict) -> float:
    """
    Simple composite for ranking: lower is better.
    Equal-weighted sum of normalised F1, F2, F3 test values.
    Returns inf if any metric is missing.
    """
    try:
        return float(row["test_F1"]) + float(row["test_F2"]) + float(row["test_F3"]) / 30.0
    except (TypeError, ValueError):
        return float("inf")


def parse_args():
    p = argparse.ArgumentParser(description="Grid search over GNN loss weights")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=512)
    p.add_argument("--patience",   type=int,   default=15)
    p.add_argument("--device",     type=str,   default="cuda")
    p.add_argument("--configs",    type=str,   default="",
                   help="Comma-separated config IDs to run (e.g. A,B,C). "
                        "Default: run all.")
    return p.parse_args()


def main():
    args = parse_args()

    run_ids = set(args.configs.upper().split(",")) if args.configs else set()
    configs_to_run = [c for c in CONFIGS
                      if not run_ids or c["config_id"] in run_ids]

    print(f"Running {len(configs_to_run)} configuration(s)  "
          f"({args.epochs} epochs each, patience={args.patience})")

    results = []
    for cfg in configs_to_run:
        row = run_config(cfg, args)
        results.append(row)
        print(f"  → Config {cfg['config_id']} done: "
              f"test F1={row.get('test_F1')}  "
              f"F2={row.get('test_F2')}  "
              f"F3={row.get('test_F3')} min  "
              f"({row['elapsed_s']}s)")

    df = pd.DataFrame(results)
    df["composite"] = df.apply(composite_score, axis=1)
    df = df.sort_values("composite")

    out_csv = ROOT / "outputs" / "grid_search_results.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n{'='*70}")
    print("Grid Search Results (ranked by composite score F1 + F2 + F3/30):")
    print(df[["config_id", "label", "alpha", "beta", "gamma", "lam",
              "test_F1", "test_F2", "test_F3", "composite"]].to_string(index=False))
    print(f"\nFull results saved to {out_csv}")

    best = df.iloc[0]
    print(f"\nBest config: {best['config_id']} ({best['label']})")
    print(f"  Re-run with:")
    print(f"  python -m src.train --alpha {best['alpha']} --beta {best['beta']} "
          f"--gamma {best['gamma']} --lam {best['lam']} "
          f"--entropy_weight {best['entropy_weight']} --epochs 100")


if __name__ == "__main__":
    main()
