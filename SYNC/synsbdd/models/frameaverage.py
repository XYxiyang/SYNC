import torch
import torch.nn as nn
from sru import SRUpp
from torch_geometric.utils import to_dense_batch


class FrameAveraging(nn.Module):

    def __init__(self):
        super(FrameAveraging, self).__init__()
        self.ops = torch.tensor([
                [i,j,k] for i in [-1,1] for j in [-1,1] for k in [-1,1]
        ]).cuda()

    def create_frame(self, X, mask):
        """
        Create a frame for each molecule.
        X: [B,N,3]
        mask: [B,N]
        """
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B,N,3]
        C = torch.bmm(X.transpose(1,2), X)  # [B,3,3] (Cov)
        _, V = torch.linalg.eigh(C.detach())  # [B,3,3]
        F_ops = self.ops.unsqueeze(1).unsqueeze(0) * V.unsqueeze(1)  # [1,8,1,3] x [B,1,3,3] -> [B,8,3,3]
        h = torch.einsum('boij,bpj->bopi', F_ops.transpose(2,3), X)  # transpose is inverse [B,8,N,3]
        h = h.view(X.size(0) * 8, X.size(1), 3)
        return h, F_ops.detach(), center

    def invert_frame(self, X, mask, F_ops, center):
        """
        Invert the frame for each molecule.
        X: [B,N,3]
        mask: [B,N]
        F_ops: [B,8,3,3]
        center: [B,3]
        """
        X = torch.einsum('boij,bopj->bopi', F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)
        return X * mask.unsqueeze(-1)

    def to_batch(self, data, batch_idx):
        """
        Convert the data to a batch.
        """
        data_batch = {}
        for k, v in data.items():
            if v.shape[0] == batch_idx.shape[0]:
                data_batch[k], data_batch[f'{k}_mask'] = to_dense_batch(v, batch_idx)
            else:
                data_batch[k] = v
        return data_batch

class FAEncoder(FrameAveraging):
    """
    Encode the binder and target molecules using frame averaging.

    """
    def __init__(self, cfg):
        super(FAEncoder, self).__init__()
        self.cfg = cfg
        self.W_bind = nn.Linear(cfg.node_feat_dim, cfg.hidden_size)
        self.W_tgt = nn.Linear(cfg.node_feat_dim, cfg.hidden_size)
        self.bind_flag = nn.Embedding(2, cfg.hidden_size)
        self.encoder = SRUpp(
            cfg.hidden_size + 3,
            cfg.hidden_size,
            cfg.hidden_size // 2,
            num_heads=cfg.n_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=True,
        )

    def forward(self, binder, target):
        binder = self.to_batch(binder, binder['batch_idx'])
        target = self.to_batch(target, target['batch_idx'])

        (
            bind_X, 
            bind_S, 
            bind_msk, 
            bind_gen_msk,
            bind_flag,
            bind_batch
        ) = (
            binder['x'], 
            binder['h'], 
            binder['x_mask'], 
            binder['gen_flag'], 
            binder['lig_flag'].long(),
            binder['batch_idx']
        )
        (
            tgt_X, 
            tgt_S, 
            tgt_msk, 
            tgt_gen_msk,
            tgt_flag,
            tgt_batch
        ) = (
            target['x'], 
            target['h'], 
            target['x_mask'], 
            target['gen_flag'], 
            target['lig_flag'].long(),
            target['batch_idx']
        )

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_S.size(1)

        mask = torch.cat([bind_msk, tgt_msk], dim=1)
        gen_mask = torch.cat([bind_gen_msk, tgt_gen_msk], dim=1)

        bind_S = self.W_bind(bind_S) + self.bind_flag(bind_flag)
        tgt_S = self.W_tgt(tgt_S) + self.bind_flag(tgt_flag)

        h_S = torch.cat([bind_S, tgt_S], dim=1)
        X = torch.cat([bind_X, tgt_X], dim=1)
        h_X, _, _ = self.create_frame(X, mask)
        h_S = h_S.unsqueeze(1).expand(-1, 8, -1, -1).reshape(B*8, N+M, -1)
        mask_fa = mask.unsqueeze(1).expand(-1, 8, -1).reshape(B*8, N+M)

        h = torch.cat([h_X, h_S], dim=-1)
        h, _, _ = self.encoder(
            h.transpose(0, 1),
            mask_pad=(~mask_fa.transpose(0, 1).bool())
        )
        h = torch.where(mask_fa.transpose(0, 1).unsqueeze(-1), h, torch.zeros_like(h))
        h = h.transpose(0, 1).view(B, 8, N+M, -1)
        return h.mean(dim=1), mask, gen_mask   # frame averaging


class FAPredictor(nn.Module):
    """
    Predict the property of the molecule.
    """
    def __init__(self, cfg):
        super(FAPredictor, self).__init__()
        self.encoder = FAEncoder(cfg)
        self.W_out = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size * 2),
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(
                cfg.hidden_size, 
                (len(FG_LABEL_KEYS)+len(PLIP_LABEL_KEYS))*4, 
                bias=False
            ),
        )

    def forward(self, binder, target):
        B = binder['batch_idx'].max().item() + 1
        h, mask, gen_mask = self.encoder(binder, target)
        h_graph = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)
        fg_prop, plip_prop = self.W_out(h_graph).split([len(FG_LABEL_KEYS)*4, len(PLIP_LABEL_KEYS)*4], dim=-1)
        fg_prop = fg_prop.view(B, len(FG_LABEL_KEYS), -1)
        plip_prop = plip_prop.view(B, len(PLIP_LABEL_KEYS), -1)
        return fg_prop, plip_prop