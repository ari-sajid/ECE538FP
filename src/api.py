"""
Inference API for the Airport Gate Scheduling GNN (Step 7).

Wraps the trained SpatioTemporalGNN in a lightweight FastAPI server that
accepts a JSON flight schedule and returns terminal assignments + predicted
delays in real time.

Start the server
----------------
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Or from the project root:
    python src/api.py

Endpoints
---------
GET  /health          Liveness check; reports model load status.
GET  /gates           List the five terminal classes and their distances.
POST /predict         Gate assignments for a batch of flights (≤ 500).

Example request (POST /predict)
--------------------------------
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{
               "flights": [
                 {
                   "carrier":    "UA",
                   "origin":     "EWR",
                   "dest":       "LAX",
                   "dep_time":   800,
                   "distance":   2475.0,
                   "fl_date":    "2025-11-15"
                 }
               ]
             }'

If no checkpoint exists the server falls back to the Greedy First-Fit
baseline (π₀) and marks each assignment with  "policy": "baseline".
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ── FastAPI / Pydantic ──────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "FastAPI is required for the inference API.\n"
        "Install with:  pip install fastapi uvicorn"
    )

# ── Project imports ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import SpatioTemporalGNN, GATE_CLASSES, NUM_GATES  # noqa: E402
from src.baseline import (                                          # noqa: E402
    GreedyFirstFit, APPROX_DIST_M, GATE_TO_IDX, IDX_TO_GATE,
)

GATE_MAP    = ROOT / "data" / "meta" / "gate_mapping.json"
CHECKPOINT  = ROOT / "outputs" / "best_model.pt"
SCHEMA_FILE = ROOT / "outputs" / "feature_schema.json"

# ---------------------------------------------------------------------------
# App and global state
# ---------------------------------------------------------------------------
app = FastAPI(
    title       = "Airport Gate Scheduling API",
    description = "Real-time terminal assignment using a trained Spatio-Temporal GNN.",
    version     = "1.0.0",
)

_model:    Optional[SpatioTemporalGNN] = None
_schema:   Optional[dict]              = None
_baseline: Optional[GreedyFirstFit]    = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model() -> bool:
    """Load GNN checkpoint and feature schema. Returns True on success."""
    global _model, _schema

    if not CHECKPOINT.exists():
        return False
    if not SCHEMA_FILE.exists():
        return False

    try:
        import torch
        with open(SCHEMA_FILE) as fh:
            _schema = json.load(fh)

        ckpt = torch.load(str(CHECKPOINT), map_location="cpu")
        _model = SpatioTemporalGNN(
            in_channels     = _schema["in_channels"],
            hidden_channels = _schema.get("hidden_channels", 128),
            num_layers      = _schema.get("num_layers", 3),
            num_heads       = _schema.get("num_heads", 4),
        )
        _model.load_state_dict(ckpt["model_state"])
        _model.eval()
        return True
    except Exception as exc:
        print(f"[API] Could not load model: {exc}")
        return False


def _get_baseline() -> GreedyFirstFit:
    global _baseline
    if _baseline is None:
        _baseline = GreedyFirstFit(str(GATE_MAP))
    return _baseline


# ---------------------------------------------------------------------------
# Feature engineering  (mirrors finalize_data.py logic)
# ---------------------------------------------------------------------------

def _build_feature_vector(flight: dict, schema: dict) -> np.ndarray:
    """
    Build a float32 feature vector for one flight, matching the column order
    stored in feature_schema.json.

    Unknown fields (weather, actual times) are filled with 0.
    """
    carrier    = flight.get("carrier", "")
    origin     = flight.get("origin",  "")
    dest       = flight.get("dest",    "")
    dep_time   = float(flight.get("dep_time",  0))
    distance   = float(flight.get("distance",  0))
    wind       = float(flight.get("wind_speed",    0.0))
    visibility = float(flight.get("visibility",    1.0))
    precip     = float(flight.get("precipitation", 0.0))

    row: dict = {
        "CRS_DEP_TIME"       : dep_time,
        "DEP_TIME"           : 0.0,
        "TAXI_OUT"           : 0.0,
        "WHEELS_OFF"         : 0.0,
        "WHEELS_ON"          : 0.0,
        "TAXI_IN"            : 0.0,
        "DISTANCE"           : distance,
        "LATE_AIRCRAFT_DELAY": 0.0,
        "tmpf"               : 0.0,
        "drct"               : 0.0,
        "alti"               : 0.0,
        "sped"               : wind,
        "p01i"               : precip,
        "vsby"               : visibility,
        # Step 3: NYC-side airport flags
        "AT_EWR"             : 1.0 if (origin == "EWR" or dest == "EWR") else 0.0,
        "AT_LGA"             : 1.0 if (origin == "LGA" or dest == "LGA") else 0.0,
    }

    # One-hot carrier
    for c in schema.get("carrier_list", []):
        row[f"OP_UNIQUE_CARRIER_{c}"] = 1.0 if c == carrier else 0.0

    # One-hot origin
    for o in schema.get("origin_list", []):
        row[f"ORIGIN_{o}"] = 1.0 if o == origin else 0.0

    # Build vector in the exact column order from training
    feature_cols = schema.get("feature_cols", list(row.keys()))
    vec = np.array([row.get(col, 0.0) for col in feature_cols], dtype=np.float32)
    return vec


def _build_edges(n: int, dep_times: list) -> tuple:
    """
    Build turnaround and congestion edges for a small inference batch.

    Turnaround : consecutive flights — flight i → i+1 (assumes input is
                 sorted by departure time per aircraft).
    Congestion : all pairs of flights departing within 15 minutes.
    """
    ta_src, ta_tgt = [], []
    cg_src, cg_tgt = [], []

    for i in range(n - 1):
        ta_src.append(i)
        ta_tgt.append(i + 1)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(dep_times[i] - dep_times[j]) <= 15:
                cg_src.append(i)
                cg_tgt.append(j)
                cg_src.append(j)
                cg_tgt.append(i)

    def _ei(src, tgt):
        if src:
            return np.array([src, tgt], dtype=np.int64)
        return np.zeros((2, 0), dtype=np.int64)

    return _ei(ta_src, ta_tgt), _ei(cg_src, cg_tgt)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class FlightInput(BaseModel):
    carrier:       str   = Field(..., example="UA",    description="IATA carrier code")
    origin:        str   = Field(..., example="EWR",   description="Origin IATA code")
    dest:          str   = Field(..., example="LAX",   description="Destination IATA code")
    dep_time:      int   = Field(..., example=800,     description="Scheduled departure (HHMM)")
    distance:      float = Field(0.0, example=2475.0,  description="Flight distance (miles)")
    fl_date:       str   = Field("",  example="2025-11-15", description="Flight date (YYYY-MM-DD)")
    wind_speed:    float = Field(0.0, description="Normalised wind speed [0-1]")
    visibility:    float = Field(1.0, description="Normalised visibility [0-1]")
    precipitation: float = Field(0.0, description="Normalised precipitation [0-1]")


class PredictRequest(BaseModel):
    flights: List[FlightInput] = Field(..., max_length=500)


class GateAssignment(BaseModel):
    flight_idx:           int
    carrier:              str
    origin:               str
    dest:                 str
    terminal:             str
    terminal_idx:         int
    predicted_delay_min:  float
    confidence:           float
    probabilities:        List[float]
    policy:               str   # "gnn" | "baseline"


class PredictResponse(BaseModel):
    assignments: List[GateAssignment]
    summary: dict


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    loaded = _load_model()
    if loaded:
        print(f"[API] GNN model loaded from {CHECKPOINT}")
    else:
        print("[API] No trained checkpoint found — using greedy baseline fallback.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "model_loaded": _model is not None,
        "checkpoint"  : str(CHECKPOINT) if CHECKPOINT.exists() else None,
        "in_channels" : _schema["in_channels"] if _schema else None,
    }


@app.get("/gates")
def list_gates():
    return {
        "gates": [
            {"idx": i, "name": g, "approx_dist_m": APPROX_DIST_M.get(g, 0)}
            for i, g in enumerate(GATE_CLASSES)
        ]
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    flights = request.flights
    n = len(flights)

    if n == 0:
        raise HTTPException(status_code=400, detail="flights list is empty")

    dep_times = [f.dep_time for f in flights]

    # ── GNN path ─────────────────────────────────────────────────────────
    if _model is not None and _schema is not None:
        try:
            import torch
            from torch_geometric.data import HeteroData

            # Build feature matrix
            X = np.stack([_build_feature_vector(f.dict(), _schema) for f in flights])

            # Pad/trim to match expected in_channels
            in_ch = _schema["in_channels"]
            if X.shape[1] < in_ch:
                X = np.pad(X, ((0, 0), (0, in_ch - X.shape[1])))
            elif X.shape[1] > in_ch:
                X = X[:, :in_ch]

            # Build mini-graph edges
            ta_ei, cg_ei = _build_edges(n, dep_times)

            data = HeteroData()
            data["flight"].x         = torch.from_numpy(X)
            data["flight"].num_nodes = n

            ei_dict = {}
            if ta_ei.shape[1] > 0:
                ei_dict[("flight", "turnaround", "flight")] = torch.from_numpy(ta_ei)
            if cg_ei.shape[1] > 0:
                ei_dict[("flight", "congestion", "flight")] = torch.from_numpy(cg_ei)

            with torch.no_grad():
                gate_logits, delay_pred = _model({"flight": data["flight"].x}, ei_dict)

            probs       = torch.softmax(gate_logits, dim=-1).numpy()
            hard_gates  = probs.argmax(axis=1)
            confidence  = probs.max(axis=1)
            delays      = delay_pred.numpy()
            policy      = "gnn"

        except Exception as exc:
            # Any inference error → fall back to baseline
            print(f"[API] GNN inference error: {exc}; falling back to baseline.")
            probs, hard_gates, confidence, delays, policy = None, None, None, None, "baseline"
    else:
        probs, hard_gates, confidence, delays, policy = None, None, None, None, "baseline"

    # ── Baseline fallback ─────────────────────────────────────────────────
    if policy == "baseline":
        bl = _get_baseline()
        bl_df = pd.DataFrame([{
            "OP_UNIQUE_CARRIER": f.carrier,
            "ORIGIN"           : f.origin,
            "DEST"             : f.dest,
            "CRS_DEP_TIME"     : f.dep_time,
            "FL_DATE"          : f.fl_date or "2025-01-01",
        } for f in flights])
        hard_gates  = bl.assign(bl_df)
        confidence  = np.ones(n) * 1.0
        delays      = np.zeros(n)
        probs       = np.zeros((n, NUM_GATES))
        for i, g in enumerate(hard_gates):
            if 0 <= g < NUM_GATES:
                probs[i, g] = 1.0

    # ── Build response ────────────────────────────────────────────────────
    assignments = []
    for i, f in enumerate(flights):
        g    = int(hard_gates[i]) if hard_gates[i] != -1 else 0
        name = IDX_TO_GATE.get(g, "UNKNOWN")
        assignments.append(GateAssignment(
            flight_idx          = i,
            carrier             = f.carrier,
            origin              = f.origin,
            dest                = f.dest,
            terminal            = name,
            terminal_idx        = g,
            predicted_delay_min = round(float(delays[i]), 2),
            confidence          = round(float(confidence[i]), 4),
            probabilities       = [round(float(p), 4) for p in probs[i]],
            policy              = policy,
        ))

    mean_dist  = float(np.mean([APPROX_DIST_M.get(a.terminal, 0) for a in assignments]))
    mean_delay = float(np.mean([max(0.0, a.predicted_delay_min) for a in assignments]))

    return PredictResponse(
        assignments = assignments,
        summary     = {
            "n_flights"                : n,
            "policy"                   : policy,
            "f2_mean_dist_m"           : round(mean_dist, 1),
            "f3_mean_predicted_delay_min": round(mean_delay, 2),
        },
    )


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
