import torch
import torch.nn as nn
from .egnn.egnn import EGNN
from .output_head import ClassifierHead
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

class Syn3dlin(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.atom_encoder = nn.Embedding(cfg.token_type, cfg.encoder.in_node_nf, padding_idx=0)
        self.edge_encoder = nn.Embedding(cfg.edge_type, cfg.encoder.in_edge_nf, padding_idx=0)
        #self.atom_encoder = nn.Linear(cfg.token_type, cfg.encoder.in_node_nf)
        #self.edge_encoder = nn.Linear(cfg.edge_type, cfg.encoder.in_edge_nf)
        self.encoder = EGNN(**cfg.encoder)
        self.head = ClassifierHead(**cfg.head)
        #nn.init.xavier_uniform_(self.model.layer.weight)
    
    def forward_(self, batch):

        
        
        node = self.atom_encoder(batch.x).squeeze()
        node = node.reshape(-1, node.shape[-1])
        edge = self.edge_encoder(batch.edge_attr).squeeze()
        node, _ = self.encoder(node, batch.positions, batch.edge_index, edge)
        mask = batch.mask 
        #x = (node.reshape(len(batch), -1, node.shape[1]) * mask).sum(1) / mask.sum(1)
        x = node.sum(0) / node.shape[0]
        return self.head(x)

    def forward(self, batch):

        knn = NearestNeighbors(n_neighbors=3)  # 找每个点的 3 个最近邻
        knn.fit(batch.positions.numpy())  # 注意：scikit-learn 要使用 numpy 数据
        distances, indices = knn.kneighbors(batch.positions.numpy())


        edges = []
        for i in range(K):
            for j in range(1, len(indices[i])):  # 排除自己，索引从 1 开始
                edges.append([i, indices[i][j]])

        # 转换为 torch.tensor 并转置成 [2, e] 格式
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        print("Edge index:", edge_index)
        
        node = F.log_softmax(batch.x, dim=-1)
        node = F.gumbel_softmax(node, hard=True)
        
        node = node @ self.atom_encoder.weight[:13]
        node = node.reshape(-1, node.shape[-1])
        edge = self.edge_encoder(batch.edge_attr).squeeze()
        node, _ = self.encoder(node, batch.positions, batch.edge_index, edge)
        mask = batch.mask 
        #x = (node.reshape(len(batch), -1, node.shape[1]) * mask).sum(1) / mask.sum(1)
        x = node.sum(0) / node.shape[0]
        return self.head(x)

        

def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index