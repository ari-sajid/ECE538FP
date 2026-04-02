"""
Spatio-Temporal GNN for Airport Gate Scheduling.

Architecture
------------
* Node type   : 'flight'  (512 k nodes, one per flight record)
* Edge type 1 : ('flight', 'turnaround', 'flight')
                Consecutive flights of the same aircraft – encodes the
                temporal propagation of delays (Schedule Stability, F3).
* Edge type 2 : ('flight', 'congestion', 'flight')
                Flights at the same airport within a 15-min departure
                window – encodes spatial runway/taxiway competition (F2).

Both edge types are handled by a HeteroConv wrapper around SAGEConv layers.
Messages from both relation types are summed (aggr='sum' at the HeteroConv
level) after each hop, a residual connection is added, and LayerNorm is
applied before dropout.

Output heads
------------
gate_head  : (N, NUM_GATES) – raw logits for terminal assignment.
             Apply softmax to obtain assignment probabilities.
delay_head : (N,)            – predicted departure delay in minutes
             (regression; may be negative for early departures).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv

# ---------------------------------------------------------------------------
# Shared gate-class constants (imported by loss.py and train.py as well)
# ---------------------------------------------------------------------------
GATE_CLASSES = [
    "EWR_Terminal_A",  # index 0
    "EWR_Terminal_B",  # index 1
    "EWR_Terminal_C",  # index 2
    "LGA_Terminal_B",  # index 3
    "LGA_Terminal_C",  # index 4
]
NUM_GATES = len(GATE_CLASSES)  # 5


class SpatioTemporalGNN(nn.Module):
    """
    Heterogeneous GNN with two task heads for gate scheduling.

    Parameters
    ----------
    in_channels : int
        Number of input node features (determined at load time from the CSV).
    hidden_channels : int
        Width of all internal layers (default 128).
    num_gates : int
        Number of terminal/gate classes (default 5 = 3 EWR + 2 LGA).
    num_layers : int
        Number of heterogeneous message-passing rounds (default 3).
    dropout : float
        Dropout probability applied after each MP round (default 0.3).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_gates: int = NUM_GATES,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection: maps raw features → hidden space
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)

        # Stack of heterogeneous message-passing layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("flight", "turnaround", "flight"): SAGEConv(
                        hidden_channels, hidden_channels, aggr="mean"
                    ),
                    ("flight", "congestion", "flight"): SAGEConv(
                        hidden_channels, hidden_channels, aggr="mean"
                    ),
                },
                aggr="sum",  # sum turnaround and congestion messages per node
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Task head: gate assignment (classification)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_gates),
        )

        # Task head: delay prediction (regression)
        self.delay_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
    ):
        """
        Parameters
        ----------
        x_dict : dict
            {'flight': FloatTensor[N, in_channels]}
        edge_index_dict : dict
            {('flight','turnaround','flight'): LongTensor[2, E_t],
             ('flight','congestion', 'flight'): LongTensor[2, E_c]}
            Edge types that are absent from a mini-batch are handled
            gracefully by HeteroConv (it skips missing relation types).

        Returns
        -------
        gate_logits : FloatTensor[N, num_gates]
            Un-normalised gate scores; apply softmax for probabilities.
        delay_pred  : FloatTensor[N]
            Predicted departure delay in minutes.
        """
        # Project raw features into hidden space
        x = F.relu(self.input_norm(self.input_proj(x_dict["flight"])))

        # Heterogeneous message passing with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x_new = conv({"flight": x}, edge_index_dict)["flight"]
            x = F.dropout(norm(x_new + x_in), p=self.dropout, training=self.training)

        gate_logits = self.gate_head(x)           # [N, num_gates]
        delay_pred = self.delay_head(x).squeeze(-1)  # [N]

        return gate_logits, delay_pred
