"""
Multi-objective loss for the airport gate-scheduling GNN.

Objective definitions (lower is better for all three)
------------------------------------------------------
F1 – Gate Constraint Loss
     Penalises the probability mass placed on terminals that are
     incompatible with the flight's airline, as defined in gate_mapping.json.

F2 – Taxiing Distance Loss
     Minimises the *expected* taxiing distance from the predicted terminal
     to the active runway, computed as:
         E[dist] = softmax(gate_logits) · precomputed_distance_vector
     where the distance vector is built once from the airport GraphML
     road networks using NetworkX shortest-path (Dijkstra).

F3 – Schedule Stability Loss
     Penalises positive predicted departure delays so that gate assignments
     which propagate delays are discouraged.

Auxiliary supervised loss (not part of the Pareto front)
---------------------------------------------------------
L_reg  – MSE between the delay predictor's output and the actual
         departure delay.  Keeps the delay head calibrated so F3 is
         meaningful.

Combined loss
-------------
    L = α·F1 + β·F2 + γ·F3 + λ·L_reg

All three objectives are differentiable with respect to gate_logits so
gradients flow back into the GNN's gate head.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-declare constants here to avoid circular imports with model.py
GATE_CLASSES: List[str] = [
    "EWR_Terminal_A",  # 0
    "EWR_Terminal_B",  # 1
    "EWR_Terminal_C",  # 2
    "LGA_Terminal_B",  # 3
    "LGA_Terminal_C",  # 4
]
NUM_GATES: int = len(GATE_CLASSES)  # 5

# Approximate (lat, lon) centroids for each terminal gate-area.
# Derived from published airport diagrams; used to locate the nearest
# road-network node in the GraphML graphs.
TERMINAL_COORDS: Dict[str, Tuple[float, float]] = {
    "EWR_Terminal_A": (40.6892, -74.1751),
    "EWR_Terminal_B": (40.6927, -74.1739),
    "EWR_Terminal_C": (40.6963, -74.1734),
    "LGA_Terminal_B": (40.7757, -73.8763),
    "LGA_Terminal_C": (40.7742, -73.8790),
}

# Approximate runway midpoints used as the taxiing destination.
RUNWAY_COORDS: Dict[str, Tuple[float, float]] = {
    "EWR": (40.6878, -74.1860),  # centre of runway complex 22L/4R
    "LGA": (40.7700, -73.8820),  # centre of runway complex 04/22
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _nearest_node(G: nx.Graph, lat: float, lon: float):
    """Return the graph node whose (y, x) coords are closest to (lat, lon)."""
    best_node, best_dist = None, float("inf")
    for node, data in G.nodes(data=True):
        try:
            d = _haversine_m(lat, lon, float(data["y"]), float(data["x"]))
        except (KeyError, ValueError):
            continue
        if d < best_dist:
            best_dist = d
            best_node = node
    return best_node


def _path_length(G: nx.Graph, src, tgt) -> float:
    """Shortest weighted path length (metres); falls back to haversine."""
    try:
        return nx.shortest_path_length(G, src, tgt, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound, nx.exception.NetworkXError):
        sd, td = G.nodes[src], G.nodes[tgt]
        return _haversine_m(
            float(sd["y"]), float(sd["x"]),
            float(td["y"]), float(td["x"]),
        )


# ---------------------------------------------------------------------------
# F1 – Gate Constraint Loss
# ---------------------------------------------------------------------------

class GateConstraintLoss(nn.Module):
    """
    Soft infeasibility penalty for gate assignments that violate airline-
    terminal constraints defined in gate_mapping.json.

    For every flight the penalty equals the total softmax probability mass
    placed on terminals the airline is not authorised to use.  Carriers
    absent from the mapping receive a zero penalty (all gates are implicitly
    allowed so the model is not penalised for unknown airlines).

    Parameters
    ----------
    gate_mapping_path : str or Path
    carrier_list : list[str]
        Ordered list of carrier codes matching the one-hot columns in the
        node-feature matrix (e.g. ['AA', 'AC', 'AI', ...]).
    """

    def __init__(self, gate_mapping_path: str, carrier_list: List[str]):
        super().__init__()
        with open(gate_mapping_path) as fh:
            mapping = json.load(fh)

        airport_order = ["EWR", "LGA"]   # axis-0 of the constraint tensor
        num_carriers = len(carrier_list)
        gate_to_idx = {g: i for i, g in enumerate(GATE_CLASSES)}

        # invalid[c, a, g] = 1  ↔  carrier c must NOT use gate g at airport a
        # Default: unknown carriers → all gates valid (mask = 0)
        invalid = torch.zeros(num_carriers, 2, NUM_GATES)

        for ap_idx, airport in enumerate(airport_order):
            if airport not in mapping:
                continue
            # Gather all authorised (carrier, gate) pairs for this airport
            authorised: Dict[int, set] = {}   # carrier_idx → set of gate indices
            for terminal, carriers in mapping[airport].items():
                gate_key = f"{airport}_{terminal}"
                if gate_key not in gate_to_idx:
                    continue
                g_idx = gate_to_idx[gate_key]
                for carrier in carriers:
                    if carrier in carrier_list:
                        c_idx = carrier_list.index(carrier)
                        authorised.setdefault(c_idx, set()).add(g_idx)

            # For carriers that appear in the mapping, mark all OTHER gates as
            # invalid (gates at the OTHER airport are also invalid for this ap).
            all_gate_indices = set(range(NUM_GATES))
            for c_idx, valid_gates in authorised.items():
                for g_idx in all_gate_indices - valid_gates:
                    invalid[c_idx, ap_idx, g_idx] = 1.0

        # Also mark cross-airport gates as invalid:
        # EWR flights (ap_idx=0) must not be assigned LGA terminals (3,4)
        # LGA flights (ap_idx=1) must not be assigned EWR terminals (0,1,2)
        ewr_gate_ids = [0, 1, 2]
        lga_gate_ids = [3, 4]
        for c_idx in range(num_carriers):
            for g in lga_gate_ids:
                invalid[c_idx, 0, g] = 1.0   # EWR flight, LGA terminal → invalid
            for g in ewr_gate_ids:
                invalid[c_idx, 1, g] = 1.0   # LGA flight, EWR terminal → invalid

        self.register_buffer("invalid", invalid)   # [C, 2, G]
        self.carrier_list = carrier_list

    def forward(
        self,
        gate_logits: torch.Tensor,   # [N, G]
        carrier_ohe: torch.Tensor,   # [N, C]  (float one-hot)
        is_lga: torch.Tensor,        # [N]     (float: 1 = LGA, 0 = EWR)
    ) -> torch.Tensor:
        """Returns mean infeasibility probability (scalar)."""
        gate_probs = torch.softmax(gate_logits, dim=-1)   # [N, G]

        # Per-flight invalid mask: einsum carrier_ohe [N,C] × invalid [C,2,G]
        # → [N, 2, G]
        per_flight_inv = torch.einsum("nc,cag->nag", carrier_ohe.float(),
                                      self.invalid)

        # Select the airport dimension (0=EWR, 1=LGA) per flight
        ap_idx = is_lga.long().clamp(0, 1)                        # [N]
        ap_idx_exp = ap_idx.view(-1, 1, 1).expand(-1, 1, NUM_GATES)
        inv_mask = per_flight_inv.gather(1, ap_idx_exp).squeeze(1)  # [N, G]

        penalty = (gate_probs * inv_mask).sum(dim=-1)  # [N]
        return penalty.mean()


# ---------------------------------------------------------------------------
# F2 – Taxiing Distance Loss
# ---------------------------------------------------------------------------

class TaxiingDistanceLoss(nn.Module):
    """
    Differentiable expected taxiing distance.

    At initialisation, the shortest network path (metres) from each of the
    five terminals to its airport's runway is computed once from the GraphML
    graphs.  During training the differentiable expected distance

        E[dist] = softmax(gate_logits) · dist_vec          [N,]

    is minimised, so gradients flow back through the gate logits.

    Parameters
    ----------
    ewr_graphml, lga_graphml : str or Path
        Paths to the OSMnx GraphML files for EWR and LGA respectively.
    """

    def __init__(self, ewr_graphml: str, lga_graphml: str):
        super().__init__()
        print("Loading EWR airport network graph…")
        G_ewr = nx.read_graphml(str(ewr_graphml))
        print("Loading LGA airport network graph…")
        G_lga = nx.read_graphml(str(lga_graphml))

        dist_vec = self._build_distance_vector(G_ewr, G_lga)
        self.register_buffer("dist_vec", dist_vec)  # [NUM_GATES]

    @staticmethod
    def _build_distance_vector(G_ewr: nx.Graph, G_lga: nx.Graph) -> torch.Tensor:
        """Pre-compute and normalise terminal→runway distances for all gates."""
        graphs = {"EWR": G_ewr, "LGA": G_lga}
        distances: List[float] = []

        for gate_name in GATE_CLASSES:
            airport = gate_name.split("_")[0]  # 'EWR' or 'LGA'
            G = graphs[airport]

            t_lat, t_lon = TERMINAL_COORDS[gate_name]
            r_lat, r_lon = RUNWAY_COORDS[airport]

            t_node = _nearest_node(G, t_lat, t_lon)
            r_node = _nearest_node(G, r_lat, r_lon)
            dist_m = _path_length(G, t_node, r_node)
            print(f"  {gate_name:25s} → runway: {dist_m:7.0f} m")
            distances.append(dist_m)

        dist_t = torch.tensor(distances, dtype=torch.float32)
        # Normalise to [0, 1] so F2 is on the same scale as F1 and F3
        dist_t = dist_t / (dist_t.max() + 1e-8)
        return dist_t  # [NUM_GATES]

    def forward(
        self,
        gate_logits: torch.Tensor,              # [N, G]
        is_at_nyc: Optional[torch.Tensor] = None,  # [N] bool/float mask
    ) -> torch.Tensor:
        """
        Expected normalised taxiing distance, averaged over active flights.

        Parameters
        ----------
        is_at_nyc : optional mask
            If provided, only flights that depart from (or arrive at) EWR/LGA
            contribute to the loss.  Others are structurally irrelevant.
        """
        gate_probs = torch.softmax(gate_logits, dim=-1)        # [N, G]
        expected = (gate_probs * self.dist_vec).sum(dim=-1)    # [N]

        if is_at_nyc is not None:
            mask = is_at_nyc.float()
            n_active = mask.sum().clamp(min=1.0)
            return (expected * mask).sum() / n_active

        return expected.mean()


# ---------------------------------------------------------------------------
# F3 – Schedule Stability Loss
# ---------------------------------------------------------------------------

class ScheduleStabilityLoss(nn.Module):
    """
    Penalise the mean predicted positive departure delay.

    ReLU ensures that early departures (delay < 0) are not penalised;
    only flights predicted to depart late contribute to the loss.
    """

    def forward(self, delay_pred: torch.Tensor) -> torch.Tensor:
        return F.relu(delay_pred).mean()


# ---------------------------------------------------------------------------
# Auxiliary: supervised delay regression loss
# ---------------------------------------------------------------------------

class DelayRegressionLoss(nn.Module):
    """
    MSE between predicted and observed departure delay (minutes).

    This is NOT part of the three-objective Pareto front; it is an auxiliary
    supervised signal that keeps the delay head calibrated so that F3 is
    a meaningful measure of schedule stability.
    """

    def forward(
        self,
        delay_pred: torch.Tensor,  # [N]
        delay_true: torch.Tensor,  # [N]
    ) -> torch.Tensor:
        return F.mse_loss(delay_pred, delay_true.float())


# ---------------------------------------------------------------------------
# Combined multi-objective loss
# ---------------------------------------------------------------------------

class MultiObjectiveLoss(nn.Module):
    """
    L = α·F1 + β·F2 + γ·F3 + λ·L_reg

    Parameters
    ----------
    gate_mapping_path : str
    ewr_graphml, lga_graphml : str
    carrier_list : list[str]
        Carrier codes in the same order as the OHE columns in the CSV.
    alpha, beta, gamma : float
        Pareto trade-off weights for F1, F2, F3.
    lam : float
        Weight of the auxiliary delay regression loss L_reg.
    """

    def __init__(
        self,
        gate_mapping_path: str,
        ewr_graphml: str,
        lga_graphml: str,
        carrier_list: List[str],
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        lam: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam = lam

        self.f1 = GateConstraintLoss(gate_mapping_path, carrier_list)
        self.f2 = TaxiingDistanceLoss(ewr_graphml, lga_graphml)
        self.f3 = ScheduleStabilityLoss()
        self.l_reg = DelayRegressionLoss()

    def forward(
        self,
        gate_logits: torch.Tensor,   # [N, 5]
        delay_pred: torch.Tensor,    # [N]
        delay_true: torch.Tensor,    # [N]
        carrier_ohe: torch.Tensor,   # [N, C]
        is_lga: torch.Tensor,        # [N]  float: 1 = LGA, 0 = EWR
        is_at_nyc: torch.Tensor,     # [N]  float: 1 = flight uses EWR or LGA
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total  : scalar – combined back-prop loss
        f1     : scalar – gate infeasibility  (for Pareto logging)
        f2     : scalar – expected taxi dist  (for Pareto logging)
        f3     : scalar – mean positive delay (for Pareto logging)
        """
        loss_f1 = self.f1(gate_logits, carrier_ohe, is_lga)
        loss_f2 = self.f2(gate_logits, is_at_nyc)
        loss_f3 = self.f3(delay_pred)
        loss_reg = self.l_reg(delay_pred, delay_true)

        total = (
            self.alpha * loss_f1
            + self.beta  * loss_f2
            + self.gamma * loss_f3
            + self.lam   * loss_reg
        )
        return total, loss_f1, loss_f2, loss_f3
