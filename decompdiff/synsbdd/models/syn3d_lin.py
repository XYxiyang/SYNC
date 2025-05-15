import torch
import torch.nn as nn
from .egnn.egnn import EGNN
from .output_head import ClassifierHead
import torch.nn.functional as F

class Syn3dlin(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.atom_encoder = nn.Embedding(158, cfg.encoder.in_node_nf, padding_idx=0)
        self.edge_encoder = nn.Embedding(6, cfg.encoder.in_edge_nf, padding_idx=0)
        #self.atom_encoder = nn.Linear(cfg.token_type, cfg.encoder.in_node_nf)
        #self.edge_encoder = nn.Linear(cfg.edge_type, cfg.encoder.in_edge_nf)
        self.encoder = EGNN(**cfg.encoder)
        self.head = ClassifierHead(**cfg.head)
        self.token_type = cfg.token_type
        self.edge_type = cfg.edge_type
        #nn.init.xavier_uniform_(self.model.layer.weight)
    
    def forward_(self, batch, batch_ligand):

        bsz = batch_ligand.max() + 1
        
        node = F.gumbel_softmax(batch['x'], hard=True)

        
        #edge = F.one_hot(batch["edge_attr"], num_classes=6).float()

        #node = node @ self.atom_encoder.weight[:, :13].T + self.atom_encoder.bias
        node = node @ self.atom_encoder.weight[:13]
        #edge = self.edge_encoder(batch["edge_attr"]).squeeze()
        node, _ = self.encoder(node, batch["positions"], batch["edge_index"],   None)
    

        node = node * batch["mask"]

        group_sum = torch.zeros((bsz, node.shape[1]), device=node.device )
        group_count = torch.zeros(bsz, device=node.device)
        
        for i in range(len(batch_ligand)):
            group_sum[batch_ligand[i]] += node[i]
            group_count[batch_ligand[i]] += 1

        group_mean = group_sum / group_count.unsqueeze(1)
        

        return self.head(group_mean)

    def forward(self, batch, batch_ligand):

        bsz = batch_ligand.max() + 1
        node = F.gumbel_softmax(batch['x'], hard=True)
        
        #edge = F.one_hot(batch["edge_attr"], num_classes=6).float()
        #edge = self.edge_encoder(batch["edge_attr"]).squeeze()

        #node = node @ self.atom_encoder.weight[:, :13].T + self.atom_encoder.bias
        node = node @ self.atom_encoder.weight[1:9]
        #node = self.atom_encoder(batch['x']).squeeze()
        node = node.reshape(-1, node.shape[-1])
        node, _ = self.encoder(node, batch['positions'], batch['edge_index'], None)
        #x = (node.reshape(batch.len, -1, node.shape[1]) * mask).sum(1) / mask.sum(1)
        #x = node.sum(0) / node.shape[0]
        group_sum = torch.zeros((bsz, node.shape[1]), device=node.device )
        group_count = torch.zeros(bsz, device=node.device)
        
        for i in range(len(batch_ligand)):
            group_sum[batch_ligand[i]] += node[i]
            group_count[batch_ligand[i]] += 1

        group_mean = group_sum / group_count.unsqueeze(1)
        return self.head(group_mean)
    
    def predict(self, batch):
        node = self.atom_encoder(batch['x']).squeeze()
        #edge = self.edge_encoder(batch['edge_attr']).squeeze()
        node = node.reshape(-1, node.shape[-1])
        #print(batch['edge_index'])
        node, _ = self.encoder(node, batch['positions'], batch['edge_index'], None)
        #mask = batch.mask 
        #x = (node.reshape(len(batch), -1, node.shape[1]) * mask).sum(1) / mask.sum(1)
        x = node.sum(0) / node.shape[0]

        return self.head(x)
    
    def evaluate(self, batch):
        node = self.atom_encoder(batch.x).squeeze()
        edge = self.edge_encoder(batch.edge_attr).squeeze()
        node = node.reshape(-1, node.shape[-1])
        node, _ = self.encoder(node, batch.positions, batch.edge_index, edge)
        #mask = batch.mask 
        #x = (node.reshape(len(batch), -1, node.shape[1]) * mask).sum(1) / mask.sum(1)
        x = node.sum(0) / node.shape[0]
        return self.head(x)
    
    def emb_predict(self, batch):
        node = self.atom_encoder(batch['x']).squeeze()
        edge = self.edge_encoder(batch['edge_attr']).squeeze()
        node = node.reshape(-1, node.shape[-1])
        node, _ = self.encoder(node, batch['positions'], batch['edge_index'], edge)
        #mask = batch.mask 
        #x = (node.reshape(len(batch), -1, node.shape[1]) * mask).sum(1) / mask.sum(1)
        x = node.sum(0) / node.shape[0]
        return self.head(x)