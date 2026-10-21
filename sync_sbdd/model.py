"""Portable wrapper around the original EGNN and classifier head."""
from pathlib import Path

import torch
from torch import nn

from .egnn import EGNN
from .output_head import ClassifierHead

DEFAULT_CONFIG = dict(token_type=158, edge_type=6, hidden_dim=128,
                      num_layers=4, head_dim=256)


class SYNC(nn.Module):
    """Bond-connected, bond-type-free classifier from Section 5.1/5.2."""

    def __init__(self, token_type=158, edge_type=6, hidden_dim=128,
                 num_layers=4, head_dim=256):
        super().__init__()
        self.config = dict(token_type=token_type, edge_type=edge_type,
                           hidden_dim=hidden_dim, num_layers=num_layers,
                           head_dim=head_dim)
        self.atom_encoder = nn.Embedding(token_type, hidden_dim, padding_idx=0)
        # Retain the empty parameter present in the original no-edge checkpoints.
        self.edge_encoder = nn.Embedding(edge_type, 0, padding_idx=0)
        self.encoder = EGNN(in_node_nf=hidden_dim, in_edge_nf=0,
                            out_node_nf=hidden_dim, hidden_nf=hidden_dim,
                            n_layers=num_layers, device="cpu")
        self.head = ClassifierHead(hidden_dim, 2, head_dim)

    def forward(self, atom_types, positions, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(len(atom_types), dtype=torch.long, device=atom_types.device)
        if not len(atom_types):
            raise ValueError("Cannot score an empty molecule.")
        node = self.atom_encoder(atom_types)
        node, _ = self.encoder(node, positions, edge_index, None)
        num_graphs = int(batch.max()) + 1
        pooled = node.new_zeros((num_graphs, node.shape[-1]))
        pooled.index_add_(0, batch, node)
        count = torch.bincount(batch, minlength=num_graphs).unsqueeze(-1)
        return self.head(pooled / count)


def load_model(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "sync-sbdd-state-v1":
        raise ValueError("Use a converted SYNC state checkpoint; see checkpoints/README.md.")
    if checkpoint.get("representation") != "aromatic-start1-bonds-noedge":
        raise ValueError("Checkpoint uses an incompatible atom/edge representation.")
    model = SYNC(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def save_model(path, model, **metadata):
    torch.save({
        **metadata,
        "format": "sync-sbdd-state-v1",
        "representation": "aromatic-start1-bonds-noedge",
        "model_config": model.config,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }, path)
