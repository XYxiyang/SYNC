import torch
import torch.nn as nn
from .egnn.egnn import EGNN
from .output_head import ClassifierHead
import torch.nn.functional as F


class Syn3dlin(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.atom_encoder = nn.Embedding(cfg.token_type, cfg.encoder.in_node_nf, padding_idx=0)
        #self.edge_encoder = nn.Embedding(cfg.edge_type, cfg.encoder.in_edge_nf, padding_idx=0)
        #self.atom_encoder = nn.Linear(cfg.token_type, cfg.encoder.in_node_nf)
        #self.edge_encoder = nn.Linear(cfg.edge_type, cfg.encoder.in_edge_nf)
        self.encoder = EGNN(**cfg.encoder)
        self.head = ClassifierHead(**cfg.head)
        #nn.init.xavier_uniform_(self.model.layer.weight)
    
    def forward(self, batch):
        node = self.atom_encoder(batch.x).squeeze()
        #edge = self.edge_encoder(batch.edge_attr).squeeze()
        node = node.reshape(-1, node.shape[-1])
        node, _ = self.encoder(node, batch.positions, batch.edge_index, None)
        mask = batch.mask 
        x = (node.reshape(len(batch), -1, node.shape[1]) * mask).sum(1) / mask.sum(1)
        
        #x = node.sum(0) / node.shape[0]
        return self.head(x)
    

    def forward_(self, batch):
        
        node = F.log_softmax(batch.x, dim=-1)
        node = F.gumbel_softmax(node, hard=True)
        
        node = node @ self.atom_encoder.weight[1:14]
        node = node.reshape(-1, node.shape[-1])
        edge = self.edge_encoder(batch.edge_attr).squeeze()
        node, _ = self.encoder(node, batch.positions, batch.edge_index, edge)
        mask = batch.mask 
        #x = (node.reshape(len(batch), -1, node.shape[1]) * mask).sum(1) / mask.sum(1)
        x = node.sum(0) / node.shape[0]
        return self.head(x)

    def predict(self, batch):
        node = self.atom_encoder(batch.x).squeeze()
        #edge = self.edge_encoder(batch.edge_attr).squeeze()
        node = node.reshape(-1, node.shape[-1])
        node, _ = self.encoder(node, batch.positions, batch.edge_index, None)
        x = node.sum(0) / node.shape[0]
        return self.head(x)
    
    def gradient(self, batch):
        node = batch.x @ self.atom_encoder.weight[:14]
        node, _ = self.encoder(node, batch.positions, batch.edge_index, None)
        x = node.sum(0) / node.shape[0]
        return self.head(x)
    
def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index