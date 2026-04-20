"""
Multi-objective loss for the airport gate-scheduling GNN.

Objective definitions (lower is better for all)
------------------------------------------------
F2 – Taxiing Distance Loss
     Minimises the expected taxiing distance from the predicted terminal
     to the active runway:
         E[dist] = softmax(gate_logits) · precomputed_distance_vector

F3 – Schedule Stability Loss (delay-risk proxy)
     Differentiable approximation of P(departure delay > 15 min):
         F3 = mean(sigmoid((delay_pred − 15) / 5))
     Gradient is non-zero for any delay_pred near 15 min; fixed scale
     avoids the need for a normalisation hyperparameter.

Auxiliary supervised loss
--------------------------
L_reg  – Huber loss between predicted and observed departure delay.
         Keeps the delay head calibrated so F3 is meaningful.

Combined loss
-------------
    L = β·F2 + γ·F3 + λ·L_reg + ε·(–H)

Gate feasibility is enforced *structurally* via GateMasker.mask_logits()
(infeasible logits → −∞ before every softmax) and is therefore never a
loss term — it is always zero by construction.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

GATE_CLASSES: List[str] = [
    "EWR_Terminal_A",  # 0
    "EWR_Terminal_B",  # 1
    "EWR_Terminal_C",  # 2
    "LGA_Terminal_B",  # 3
    "LGA_Terminal_C",  # 4
]
NUM_GATES: int = len(GATE_CLASSES)

# Approximate taxi distances (metres) from each terminal to its airport's runway.
# These match the values used in baseline.py for apples-to-apples comparison.
APPROX_DIST_M: Dict[str, int] = {
    "EWR_Terminal_A": 810,
    "EWR_Terminal_B": 1140,
    "EWR_Terminal_C": 1380,
    "LGA_Terminal_B": 660,
    "LGA_Terminal_C": 920,
}

# Approximate gate counts per terminal (from BTS/FAA airport facility data).
# Used by SafeOverridePolicy for terminal-specific capacity repair.
TERMINAL_CAPACITY: Dict[str, int] = {
    "EWR_Terminal_A": 25,   # mixed-use domestic
    "EWR_Terminal_B": 22,   # international
    "EWR_Terminal_C": 45,   # United hub
    "LGA_Terminal_B": 35,   # new Delta/AA terminal
    "LGA_Terminal_C": 12,   # smaller concourse
}



# ---------------------------------------------------------------------------
# Gate Masker — structural feasibility enforcement (not a loss)
# ---------------------------------------------------------------------------

class GateMasker(nn.Module):
    """
    Enforces airline-terminal constraints by masking infeasible gate logits
    to −∞ before every softmax.  This guarantees zero gate violations by
    construction — no loss term is needed.

    Parameters
    ----------
    gate_mapping_path : str or Path
    carrier_list : list[str]
    """

    def __init__(self, gate_mapping_path: str, carrier_list: List[str]):
        super().__init__()
        with open(gate_mapping_path) as fh:
            mapping = json.load(fh)

        airport_order = ["EWR", "LGA"]
        num_carriers = len(carrier_list)
        gate_to_idx = {g: i for i, g in enumerate(GATE_CLASSES)}

        invalid = torch.zeros(num_carriers, 2, NUM_GATES)

        for ap_idx, airport in enumerate(airport_order):
            if airport not in mapping:
                continue
            authorised: Dict[int, set] = {}
            for terminal, carriers in mapping[airport].items():
                gate_key = f"{airport}_{terminal}"
                if gate_key not in gate_to_idx:
                    continue
                g_idx = gate_to_idx[gate_key]
                for carrier in carriers:
                    if carrier in carrier_list:
                        c_idx = carrier_list.index(carrier)
                        authorised.setdefault(c_idx, set()).add(g_idx)

            all_gate_indices = set(range(NUM_GATES))
            for c_idx, valid_gates in authorised.items():
                for g_idx in all_gate_indices - valid_gates:
                    invalid[c_idx, ap_idx, g_idx] = 1.0

        # Cross-airport: EWR flights must not use LGA terminals and vice-versa
        ewr_gate_ids = [0, 1, 2]
        lga_gate_ids = [3, 4]
        for c_idx in range(num_carriers):
            for g in lga_gate_ids:
                invalid[c_idx, 0, g] = 1.0
            for g in ewr_gate_ids:
                invalid[c_idx, 1, g] = 1.0

        self.register_buffer("invalid", invalid)
        self.carrier_list = carrier_list

    def _per_flight_invalid_mask(
        self,
        carrier_ohe: torch.Tensor,  # [N, C]
        is_lga: torch.Tensor,       # [N]
    ) -> torch.Tensor:              # [N, G]
        per_flight_inv = torch.einsum("nc,cag->nag", carrier_ohe.float(), self.invalid)
        ap_idx     = is_lga.long().clamp(0, 1)
        ap_idx_exp = ap_idx.view(-1, 1, 1).expand(-1, 1, NUM_GATES)
        return per_flight_inv.gather(1, ap_idx_exp).squeeze(1)

    def mask_logits(
        self,
        gate_logits: torch.Tensor,  # [N, G]
        carrier_ohe: torch.Tensor,  # [N, C]
        is_lga: torch.Tensor,       # [N]
    ) -> torch.Tensor:              # [N, G]
        """Set infeasible gate logits to −∞ (hard constraint, differentiable)."""
        inv_mask = self._per_flight_invalid_mask(carrier_ohe, is_lga)
        valid    = inv_mask < 0.5

        # Fallback: if every gate is masked for a flight, allow all gates
        # (avoids softmax NaN for carriers with no valid terminal at this airport)
        has_valid = valid.any(dim=-1, keepdim=True)
        valid     = valid | ~has_valid

        neg_inf = torch.full_like(gate_logits, float("-inf"))
        return torch.where(valid, gate_logits, neg_inf)


# ---------------------------------------------------------------------------
# F2 – Taxiing Distance Loss
# ---------------------------------------------------------------------------

class TaxiingDistanceLoss(nn.Module):
    """
    Differentiable expected taxiing distance.

    Uses APPROX_DIST_M (hardcoded realistic values in metres), matching
    the distances used by baseline.py for direct apples-to-apples comparison.
    Multiply the returned normalised loss by 1380.0 to recover metres.
    """

    def __init__(self):
        super().__init__()
        raw = torch.tensor(
            [APPROX_DIST_M[g] for g in GATE_CLASSES], dtype=torch.float32
        )
        self.register_buffer("dist_vec", raw / (raw.max() + 1e-8))

    def forward(
        self,
        gate_logits: torch.Tensor,
        is_at_nyc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate_probs = torch.softmax(gate_logits, dim=-1)
        expected   = (gate_probs * self.dist_vec).sum(dim=-1)

        if is_at_nyc is not None:
            mask     = is_at_nyc.float()
            n_active = mask.sum().clamp(min=1.0)
            return (expected * mask).sum() / n_active

        return expected.mean()


# ---------------------------------------------------------------------------
# F3 – Schedule Stability Loss (delay-risk proxy)
# ---------------------------------------------------------------------------

class ScheduleStabilityLoss(nn.Module):
    """
    Differentiable approximation of P(departure delay > 15 min).

        F3 = mean(sigmoid((delay_pred − 15) / 5))

    The sigmoid "temperature" of 5 min gives a soft threshold: flights
    predicted at 15 min delay contribute 0.5, well below contribute ~0,
    well above contribute ~1.  Output is always in (0, 1).
    """

    def forward(self, delay_pred: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((delay_pred - 15.0) / 5.0).mean()


# ---------------------------------------------------------------------------
# Entropy regularisation — prevents gate-head mode collapse
# ---------------------------------------------------------------------------

class EntropyRegularization(nn.Module):
    """
    Entropy penalty: minimising H pushes the model toward confident,
    peaked gate assignments rather than uniform distributions.

    Loss = mean_entropy(softmax(gate_logits))  [higher = more uniform = worse]
    """

    def forward(self, gate_logits: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(gate_logits, dim=-1)
        probs     = torch.softmax(gate_logits, dim=-1)
        log_probs_safe = log_probs.clamp(min=-100.0)
        entropy   = -(probs * log_probs_safe).sum(dim=-1).mean()
        return entropy   # minimising → peaked (confident) predictions


# ---------------------------------------------------------------------------
# Load-balancing regularisation — discourages temporal overloading
# ---------------------------------------------------------------------------

class LoadBalancingLoss(nn.Module):
    """
    Penalise high variance of expected gate load within each 30-min
    departure bucket in the batch.  Differentiable via gate_probs.

    NOTE: Use a small delta (≤ 0.05) or 0.0.  Minimising load std pushes
    gate_probs toward uniform (max-entropy = min-variance), which fights the
    F2 gradient.  A better future design would penalise load ABOVE a
    per-terminal capacity threshold rather than minimising variance.
    """

    def forward(
        self,
        gate_probs: torch.Tensor,   # [N, G]  after masked softmax
        dep_bucket: torch.Tensor,   # [N]     CRS_DEP_TIME // 30
    ) -> torch.Tensor:
        unique_buckets = dep_bucket.unique()
        stds = []
        for b in unique_buckets:
            mask = dep_bucket == b
            if mask.sum() < 2:
                continue
            expected_load = gate_probs[mask].sum(dim=0)  # [G]
            stds.append(expected_load.std())
        if not stds:
            return gate_probs.new_tensor(0.0)
        return torch.stack(stds).mean()


# ---------------------------------------------------------------------------
# Auxiliary: supervised delay regression loss
# ---------------------------------------------------------------------------

class DelayRegressionLoss(nn.Module):
    """Huber loss between predicted and observed departure delay (normalised)."""

    def forward(
        self,
        delay_pred: torch.Tensor,
        delay_true: torch.Tensor,
        delay_scale: float = 30.0,
    ) -> torch.Tensor:
        pred_s = delay_pred / delay_scale
        true_s = delay_true.float() / delay_scale
        return F.huber_loss(pred_s, true_s, delta=1.0)


# ---------------------------------------------------------------------------
# Combined multi-objective loss
# ---------------------------------------------------------------------------

class MultiObjectiveLoss(nn.Module):
    """
    L = β·F2 + γ·F3 + λ·L_reg + ε·(–H)

    Gate feasibility is enforced structurally (GateMasker) — not as a loss.

    Parameters
    ----------
    gate_mapping_path : str
    ewr_graphml, lga_graphml : str
    carrier_list : list[str]
    alpha : float
        Retained for CLI backward-compatibility; ignored internally.
    beta, gamma : float
        Pareto trade-off weights for F2 and F3.
    lam : float
        Weight of the auxiliary delay regression loss.
    delay_scale : float
        Reference delay (minutes) used to normalise L_reg.
    entropy_weight : float
        Weight of the entropy penalty –H.
    """

    def __init__(
        self,
        gate_mapping_path: str,
        carrier_list: List[str],
        alpha: float = 0.0,   # kept for CLI compat; not used
        beta: float = 1.0,
        gamma: float = 1.0,
        lam: float = 0.1,
        delay_scale: float = 30.0,
        entropy_weight: float = 0.0,
        delta: float = 0.5,
        # Legacy graphml params accepted but ignored (distances now from APPROX_DIST_M)
        ewr_graphml: str = "",
        lga_graphml: str = "",
    ):
        super().__init__()
        self.beta           = beta
        self.gamma          = gamma
        self.lam            = lam
        self.delay_scale    = delay_scale
        self.entropy_weight = entropy_weight
        self.delta          = delta

        self.masker  = GateMasker(gate_mapping_path, carrier_list)
        self.f2      = TaxiingDistanceLoss()
        self.f3      = ScheduleStabilityLoss()
        self.l_reg   = DelayRegressionLoss()
        self.entropy = EntropyRegularization()
        self.lb      = LoadBalancingLoss()

    def forward(
        self,
        gate_logits: torch.Tensor,              # [N, 5]
        delay_pred: torch.Tensor,               # [N]
        delay_true: torch.Tensor,               # [N]
        carrier_ohe: torch.Tensor,              # [N, C]
        is_lga: torch.Tensor,                   # [N]  float: 1 = LGA, 0 = EWR
        is_at_nyc: torch.Tensor,                # [N]  float: 1 = flight uses EWR or LGA
        dep_bucket: Optional[torch.Tensor] = None,  # [N]  CRS_DEP_TIME // 30
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total  : scalar – combined back-prop loss
        f2     : scalar – expected normalised taxi distance  [0, 1]
        f3     : scalar – delay-risk proxy P(delay > 15 min) [0, 1]
        f3_raw : scalar – mean positive delay in raw minutes (logging only)
        """
        # Structural feasibility: infeasible gate logits → −∞
        masked_logits = self.masker.mask_logits(gate_logits, carrier_ohe, is_lga)
        gate_probs    = F.softmax(masked_logits, dim=-1)

        loss_f2    = self.f2(masked_logits, is_at_nyc)
        loss_f3    = self.f3(delay_pred)
        loss_reg   = self.l_reg(delay_pred, delay_true, self.delay_scale)
        loss_ent   = self.entropy(masked_logits)
        loss_lb    = (self.lb(gate_probs, dep_bucket)
                      if dep_bucket is not None
                      else masked_logits.new_tensor(0.0))

        total = (
            self.beta             * loss_f2
            + self.gamma          * loss_f3
            + self.lam            * loss_reg
            + self.entropy_weight * loss_ent
            + self.delta          * loss_lb
        )

        f3_raw = F.relu(delay_pred).mean()
        return total, loss_f2, loss_f3, f3_raw
