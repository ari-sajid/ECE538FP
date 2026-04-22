"""
Multi-objective loss for the EWR gate-scheduling GNN.

Signal decomposition
--------------------
    T_sim = T_taxi + T_queue

    T_taxi  = d(assigned_gate_class) / v        deterministic physics
    T_queue = SoftCongestionLoss proxy           differentiable; replaces hard sim during training

Training loss (lower is better for all)
----------------------------------------
    L = β·L_taxi  +  λ·L_cong  +  γ·L_turn

    L_taxi   — differentiable expected taxi time  E[T_taxi]  (TaxiingDistanceLoss)
    L_cong   — differentiable congestion proxy    E[T_queue_soft]  (SoftCongestionLoss)
    L_turn   — turnaround smoothness regulariser  (TurnaroundSmoothnessLoss)

Gate feasibility is enforced *structurally* via GateMasker.mask_logits()
(infeasible logits → −∞ before every softmax) — never a loss term.

Evaluation uses the hard M/D/K simulation in baseline.py (simulate_queuing_f3).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

GATE_CLASSES: List[str] = [
    "EWR_A_Wide",    # 0 — Terminal A widebody gates
    "EWR_A_Narrow",  # 1 — Terminal A narrowbody gates
    "EWR_B_Narrow",  # 2 — Terminal B narrowbody gates
    "EWR_C_Wide",    # 3 — Terminal C widebody gates
    "EWR_C_Narrow",  # 4 — Terminal C narrowbody gates
]
NUM_GATES: int = len(GATE_CLASSES)  # 5

# Approximate taxi distances (metres) from each gate class to EWR's active runway.
APPROX_DIST_M: Dict[str, int] = {
    "EWR_A_Wide":   810,
    "EWR_A_Narrow": 810,
    "EWR_B_Narrow": 1140,
    "EWR_C_Wide":   1380,
    "EWR_C_Narrow": 1380,
}

# Ground taxi speed used to convert distance → time (conservative 12 km/h)
TAXI_SPEED_M_PER_MIN: float = 200.0

# Total NB-equivalent slots per gate class
GATE_SLOTS: Dict[str, int] = {
    "EWR_A_Wide":   8,   # 4 physical gates × 2 slots each
    "EWR_A_Narrow": 29,
    "EWR_B_Narrow": 9,
    "EWR_C_Wide":   4,   # 2 physical gates × 2 slots each
    "EWR_C_Narrow": 55,
}

# True for gate-class indices that are narrow (cannot serve widebody aircraft)
IS_NARROW_GATE: List[bool] = [False, True, True, False, True]  # indices 0-4


# ---------------------------------------------------------------------------
# Gate Masker — structural feasibility enforcement (not a loss)
# ---------------------------------------------------------------------------

# Terminal → gate class indices mapping (for carrier auth lookup)
_TERMINAL_TO_GATE_INDICES: Dict[str, List[int]] = {
    "Terminal_A": [0, 1],  # EWR_A_Wide, EWR_A_Narrow
    "Terminal_B": [2],     # EWR_B_Narrow
    "Terminal_C": [3, 4],  # EWR_C_Wide, EWR_C_Narrow
}


class GateMasker(nn.Module):
    """
    Enforces airline-terminal + aircraft-type constraints by masking infeasible
    gate logits to −∞ before every softmax.  EWR-only.
    """

    def __init__(self, gate_mapping_path: str, carrier_list: List[str]):
        super().__init__()
        with open(gate_mapping_path) as fh:
            mapping = json.load(fh)

        num_carriers = len(carrier_list)

        # invalid[carrier_idx, gate_idx] = 1 if carrier cannot use that gate class
        invalid = torch.zeros(num_carriers, NUM_GATES)

        ewr_mapping = mapping.get("EWR", {})
        authorised: Dict[int, set] = {}
        for terminal_name, carriers in ewr_mapping.items():
            gate_indices = _TERMINAL_TO_GATE_INDICES.get(terminal_name, [])
            for carrier in carriers:
                if carrier in carrier_list:
                    c_idx = carrier_list.index(carrier)
                    authorised.setdefault(c_idx, set()).update(gate_indices)

        all_gate_indices = set(range(NUM_GATES))
        for c_idx, valid_gates in authorised.items():
            for g_idx in all_gate_indices - valid_gates:
                invalid[c_idx, g_idx] = 1.0

        self.register_buffer("carrier_invalid", invalid)   # [C, G]
        # Boolean mask: True for narrow gate classes
        narrow_mask = torch.tensor(IS_NARROW_GATE, dtype=torch.bool)
        self.register_buffer("narrow_gate_mask", narrow_mask)  # [G]
        self.carrier_list = carrier_list

    def mask_logits(
        self,
        gate_logits: torch.Tensor,   # [N, G]
        carrier_ohe: torch.Tensor,   # [N, C]
        is_widebody: torch.Tensor,   # [N]  float: 1 = widebody
    ) -> torch.Tensor:               # [N, G]
        """Set infeasible gate logits to −∞ (hard constraint, differentiable)."""
        # Carrier-level infeasibility: [N, C] @ [C, G] → [N, G]
        carrier_inv = carrier_ohe.float() @ self.carrier_invalid
        valid = carrier_inv < 0.5

        # Widebody aircraft cannot use narrow gate classes
        narrow = self.narrow_gate_mask.unsqueeze(0)        # [1, G]
        is_wide = is_widebody.bool().unsqueeze(1)          # [N, 1]
        valid = valid & ~(is_wide & narrow)

        # Fallback: if every gate is masked for a flight, allow all gates
        valid = valid | ~valid.any(dim=-1, keepdim=True)

        return torch.where(valid, gate_logits, torch.full_like(gate_logits, float("-inf")))


# ---------------------------------------------------------------------------
# L_taxi — Differentiable expected taxi time  E[T_taxi]
# ---------------------------------------------------------------------------

class TaxiingDistanceLoss(nn.Module):
    """
    E[T_taxi] = softmax(gate_logits) · time_vec
    Returns expected taxi time in minutes.
    """

    def __init__(self):
        super().__init__()
        raw = torch.tensor(
            [APPROX_DIST_M[g] for g in GATE_CLASSES], dtype=torch.float32
        )
        self.register_buffer("time_vec", raw / TAXI_SPEED_M_PER_MIN)

    def forward(
        self,
        gate_logits: torch.Tensor,
        is_at_ewr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate_probs = torch.softmax(gate_logits, dim=-1)
        expected   = (gate_probs * self.time_vec).sum(dim=-1)

        if is_at_ewr is not None:
            mask     = is_at_ewr.float()
            n_active = mask.sum().clamp(min=1.0)
            return (expected * mask).sum() / n_active

        return expected.mean()


# ---------------------------------------------------------------------------
# L_cong — Differentiable soft congestion proxy  E[T_queue_soft]
# ---------------------------------------------------------------------------

class SoftCongestionLoss(nn.Module):
    """
    Differentiable proxy for queueing delay, weighted by aircraft slot consumption:

        E[T_queue_soft] = (1/M) Σ_i Σ_j exp(-|t_i - t_j| / τ) × (p_i · p_j) × units_i × units_j

    where p_i · p_j = dot(gate_probs_i, gate_probs_j) = P(same gate class).
    Widebody flights (units=2) contribute 4× more congestion than narrowbodies (units=1).

    Parameters
    ----------
    tau : float
        Temporal kernel bandwidth in minutes (default 30).
    """

    def __init__(self, tau: float = 30.0):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        gate_probs: torch.Tensor,      # [N, G]
        dep_time_min: torch.Tensor,    # [N]
        is_at_ewr: torch.Tensor,       # [N]  float: 1 = EWR flight
        aircraft_slots: torch.Tensor,  # [N]  float: 2.0 = widebody, 1.0 = narrowbody
    ) -> torch.Tensor:
        ewr = is_at_ewr.bool()
        if ewr.sum() < 2:
            return gate_probs.new_tensor(0.0)
        p     = gate_probs[ewr]                                  # [M, G]
        t     = dep_time_min[ewr].float()                        # [M]
        units = aircraft_slots[ewr].float()                      # [M]
        dt    = (t.unsqueeze(0) - t.unsqueeze(1)).abs()          # [M, M]
        w     = torch.exp(-dt / self.tau)                        # [M, M]
        st    = p @ p.T                                          # [M, M] same-class prob
        # Weight by product of slot sizes: heavier aircraft create more congestion
        u_sq  = units.unsqueeze(0) * units.unsqueeze(1)          # [M, M]
        return (w * u_sq * st).mean()


# ---------------------------------------------------------------------------
# L_turn — Turnaround smoothness regulariser
# ---------------------------------------------------------------------------

class TurnaroundSmoothnessLoss(nn.Module):
    """
    Symmetric squared-difference regulariser for consecutive-aircraft pairs.
    Flights can recover delay after turnaround — NOT a monotonic constraint.
    """

    def forward(
        self,
        delay_pred: torch.Tensor,
        turn_src: torch.Tensor,
        turn_dst: torch.Tensor,
    ) -> torch.Tensor:
        if turn_src.numel() == 0:
            return delay_pred.new_tensor(0.0)
        return ((delay_pred[turn_src] - delay_pred[turn_dst]) ** 2).mean()


# ---------------------------------------------------------------------------
# Combined multi-objective loss
# ---------------------------------------------------------------------------

class MultiObjectiveLoss(nn.Module):
    """
    L = β·L_taxi  +  λ·L_cong  +  γ·L_turn

    Gate feasibility enforced structurally via GateMasker (carrier auth + aircraft type).

    Parameters
    ----------
    gate_mapping_path : str
    carrier_list : list[str]
    beta : float   — L_taxi weight
    lam  : float   — L_cong weight
    gamma: float   — L_turn weight
    tau  : float   — SoftCongestionLoss bandwidth (minutes)
    """

    def __init__(
        self,
        gate_mapping_path: str,
        carrier_list: List[str],
        beta: float = 1.0,
        lam: float = 0.5,
        gamma: float = 0.05,
        tau: float = 30.0,
        # Legacy params accepted but ignored
        alpha: float = 0.0,
        delta: float = 0.0,
        entropy_weight: float = 0.0,
        delay_scale: float = 30.0,
        ewr_graphml: str = "",
        lga_graphml: str = "",
    ):
        super().__init__()
        self.beta  = beta
        self.lam   = lam
        self.gamma = gamma

        self.masker = GateMasker(gate_mapping_path, carrier_list)
        self.f2     = TaxiingDistanceLoss()
        self.cong   = SoftCongestionLoss(tau=tau)
        self.turn   = TurnaroundSmoothnessLoss()

    def forward(
        self,
        gate_logits: torch.Tensor,    # [N, 5]
        delay_pred: torch.Tensor,     # [N]
        delay_true: torch.Tensor,     # [N]  (unused in loss; kept for compat)
        carrier_ohe: torch.Tensor,    # [N, C]
        is_at_ewr: torch.Tensor,      # [N]  float: 1 = EWR flight
        is_widebody: torch.Tensor,    # [N]  float: 1 = widebody aircraft
        dep_time_min: torch.Tensor,   # [N]  departure time in minutes
        turn_src: torch.Tensor,       # [E]
        turn_dst: torch.Tensor,       # [E]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total     : scalar
        loss_taxi : scalar – expected taxi time (minutes)
        loss_cong : scalar – soft congestion proxy
        loss_turn : scalar – turnaround smoothness
        """
        masked_logits = self.masker.mask_logits(gate_logits, carrier_ohe, is_widebody)
        gate_probs    = F.softmax(masked_logits, dim=-1)

        aircraft_slots = torch.where(
            is_widebody.bool(),
            gate_probs.new_tensor(2.0),
            gate_probs.new_tensor(1.0),
        )

        loss_taxi = self.f2(masked_logits, is_at_ewr)
        loss_cong = self.cong(gate_probs, dep_time_min, is_at_ewr, aircraft_slots)
        loss_turn = self.turn(delay_pred, turn_src, turn_dst)

        total = (
            self.beta  * loss_taxi
            + self.lam   * loss_cong
            + self.gamma * loss_turn
        )
        return total, loss_taxi, loss_cong, loss_turn


# ---------------------------------------------------------------------------
# TAXI_OUT noise calibration helper (called once; not used during training)
# ---------------------------------------------------------------------------

def calibrate_noise_sigma(
    taxi_out_arr: "np.ndarray",
    assignments: "np.ndarray",
    unassigned_val: int = -1,
) -> float:
    """std(TAXI_OUT − T_taxi_physics) for assigned EWR flights."""
    import numpy as np
    valid = assignments != unassigned_val
    if valid.sum() == 0:
        return 0.0
    t_physics = np.array(
        [APPROX_DIST_M[GATE_CLASSES[int(a)]] / TAXI_SPEED_M_PER_MIN
         for a in assignments[valid]],
        dtype=np.float32,
    )
    taxi_out_valid = np.asarray(taxi_out_arr, dtype=np.float32)[valid]
    return float(np.nanstd(taxi_out_valid - t_physics))
