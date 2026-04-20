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

Message-passing layers use Graph Attention (GATConv) so the model can
learn to weight neighbours differently – e.g. a connecting flight that is
already running late should receive more attention than an on-time one.
Multi-head attention (default 4 heads) is concatenated then normalised,
keeping the hidden dimension constant across all layers.

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
from torch_geometric.nn import GATConv, HeteroConv

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
    Heterogeneous Graph Attention Network with two task heads.

    Parameters
    ----------
    in_channels : int
        Number of input node features (determined at load time from the CSV).
    hidden_channels : int
        Width of all internal layers (default 128).
        Must be divisible by num_heads.
    num_gates : int
        Number of terminal/gate classes (default 5 = 3 EWR + 2 LGA).
    num_layers : int
        Number of heterogeneous message-passing rounds (default 3).
    dropout : float
        Dropout applied inside GATConv attention and after each MP round
        (default 0.3).
    num_heads : int
        Number of attention heads per GATConv layer (default 4).
        Each head operates on hidden_channels // num_heads dimensions;
        outputs are concatenated back to hidden_channels.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_gates: int = NUM_GATES,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_heads: int = 4,
    ):
        super().__init__()
        assert hidden_channels % num_heads == 0, (
            f"hidden_channels ({hidden_channels}) must be divisible by "
            f"num_heads ({num_heads})"
        )
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        head_dim = hidden_channels // num_heads

        # Input projection: maps raw features → hidden space
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)

        # Stack of heterogeneous graph-attention layers
        # GATConv(in, head_dim, heads=H, concat=True) → output size = H * head_dim
        # = hidden_channels, so residual connections are dimension-preserving.
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("flight", "turnaround", "flight"): GATConv(
                        hidden_channels, head_dim,
                        heads=num_heads, concat=True,
                        dropout=dropout, add_self_loops=False,
                    ),
                    ("flight", "congestion", "flight"): GATConv(
                        hidden_channels, head_dim,
                        heads=num_heads, concat=True,
                        dropout=dropout, add_self_loops=False,
                    ),
                    ("flight", "same_terminal", "flight"): GATConv(
                        hidden_channels, head_dim,
                        heads=num_heads, concat=True,
                        dropout=dropout, add_self_loops=False,
                    ),
                },
                aggr="sum",
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
            Edge types absent from a mini-batch are skipped by HeteroConv.

        Returns
        -------
        gate_logits : FloatTensor[N, num_gates]
            Un-normalised gate scores; apply softmax for probabilities.
        delay_pred  : FloatTensor[N]
            Predicted departure delay in minutes.
        """
        # Project raw features into hidden space
        x = F.relu(self.input_norm(self.input_proj(x_dict["flight"])))

        # Heterogeneous graph-attention message passing with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x_new = conv({"flight": x}, edge_index_dict)["flight"]
            x = F.dropout(norm(x_new + x_in), p=self.dropout, training=self.training)

        gate_logits = self.gate_head(x)              # [N, num_gates]
        delay_pred  = self.delay_head(x).squeeze(-1) # [N]

        return gate_logits, delay_pred
