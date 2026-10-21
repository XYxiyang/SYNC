import torch
import torch.nn as nn
#from .egnn.egnn import EGNN
from .output_head import ClassifierHead
import torch.nn.functional as F
from egnn_pytorch import EGNN

class Syn3dDense(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        #self.atom_encoder = nn.Embedding(cfg.token_type, cfg.encoder.in_node_nf, padding_idx=0)
        #self.edge_encoder = nn.Embedding(cfg.edge_type, cfg.encoder.in_edge_nf, padding_idx=0)
        self.atom_encoder = nn.Linear(cfg.token_type, cfg.encoder.in_node_nf)
        #self.postion_encoder = nn.Linear(3, cfg.encoder.in_node_nf)
        #self.adj_encoder = nn.Linear(1, cfg.edge_type)
        #self.edge_encoder = nn.Linear(cfg.edge_type, cfg.encoder.in_edge_nf)
        self.encoder_layers = [EGNN(dim=cfg.encoder.in_node_nf, edge_dim=1) for _ in range(cfg.encoder.n_layers)]
        self.head = ClassifierHead(**cfg.head)
        #nn.init.xavier_uniform_(self.model.layer.weight)

    def to(self, device):
        self.atom_encoder = self.atom_encoder.to(device)
        #self.postion_encoder = self.postion_encoder.to(device)
        #elf.adj_encoder = self.adj_encoder.to(device)
        #self.edge_encoder = self.edge_encoder.to(device)
        self.encoder_layers = [layer.to(device) for layer in self.encoder_layers]
        self.head = self.head.to(device)
    
    def forward(self, batch):

        node = self.atom_encoder(batch.x).squeeze()
        #pos = self.postion_encoder(batch.positions).squeeze()
        #all_feature = torch.cat([node, pos], dim=-1)
        #adj_matrice = torch.matmul(all_feature, all_feature.transpose(2, 1))

        #node = node.reshape(-1, node.shape[-1])
        #edge = self.edge_encoder(batch.edge_attr).squeeze()
        #adj_matrice_feat = self.adj_encoder(F.sigmoid((adj_matrice.unsqueeze(-1))))
        #adj_matrice_feat = adj_matrice_feat * batch.adj_mask.unsqueeze(-1)

        pos = batch.positions
        for layer in self.encoder_layers:
            node, pos = layer(node, pos, batch.adj_matrice.unsqueeze(-1))
        
        #node, _ = self.encoder(node, batch.positions, adj_matrice)
        mask = batch.node_mask 
        x = (node * mask).sum(1) / mask.sum(1)
        #x = node.sum(0) / node.shape[0]
        return self.head(x)


