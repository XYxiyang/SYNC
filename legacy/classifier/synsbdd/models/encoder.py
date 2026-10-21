'''
Modified form https://github.com/luost26/ChemProjector.git
'''
import abc
from typing import TYPE_CHECKING

import torch
from torch import nn


from synsbdd.models.transformer.graph_transformer import GraphTransformer
from synsbdd.models.transformer.positional_encoding import PositionalEncoding


class BaseEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]: ...

    @property
    @abc.abstractmethod
    def dim(self) -> int: ...

    if TYPE_CHECKING:

        def __call__(self, batch) -> tuple[torch.Tensor, torch.Tensor]: ...


class SMILESEncoder(nn.Module):
    def __init__(
        self,
        num_token_types: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        pe_max_len: int,
    ):
        super().__init__()
        self.dim = d_model
        self.smiles_emb = nn.Embedding(num_token_types, d_model, padding_idx=0)
        self.pe_enc = PositionalEncoding(
            d_model=d_model,
            max_len=pe_max_len,
        )
        self.enc = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.smiles_emb(batch)
        h = self.pe_enc(x)
        padding_mask = batch == 0  # the positions with the value of True will be ignored
        out = self.enc(h, src_key_padding_mask=padding_mask)
        return out, padding_mask


class GraphEncoder(nn.Module):
    def __init__(
        self,
        num_atom_classes: int,
        num_bond_classes: int,
        dim: int,
        depth: int,
        dim_head: int,
        edge_dim: int,
        heads: int,
        rel_pos_emb: bool,
        output_norm: bool,
    ):
        super().__init__()
        self._dim = dim
        self.atom_emb = nn.Embedding(num_atom_classes + 1, dim, padding_idx=0)
        self.bond_emb = nn.Embedding(num_bond_classes + 1, edge_dim, padding_idx=0)
        self.enc = GraphTransformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            edge_dim=edge_dim,
            heads=heads,
            rel_pos_emb=rel_pos_emb,
            output_norm=output_norm,
        )


    def forward(self, atoms, bonds, mask) -> tuple[torch.Tensor, torch.Tensor]:


        atom_emb = self.atom_emb(atoms)
        bond_emb = self.bond_emb(bonds)
        node, _ = self.enc(nodes=atom_emb, edges=bond_emb, mask=mask)
        return node, mask
