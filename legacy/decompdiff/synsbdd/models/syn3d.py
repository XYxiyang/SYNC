import torch
import torch.nn as nn
from .egnn.egnn import EGNN
from .output_head import ClassifierHead


class Syn3d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.atom_encoder = nn.Embedding(512 * 9 + 1, cfg.encoder.in_node_nf, padding_idx=0)
        self.edge_encoder = nn.Embedding(512 * 3 + 1, cfg.encoder.in_edge_nf, padding_idx=0)
        self.encoder = EGNN(**cfg.encoder)
        self.head = ClassifierHead(**cfg.head)
        #nn.init.xavier_uniform_(self.model.layer.weight)
    
    def forward(self, batch):

        node = batch.x @ self.atom_encoder.weight
        edge = batch.edge_attr @ self.edge_encoder.weight
        node, _ = self.encoder(node, batch.positions, batch.edge_index, edge)
        mask = batch.mask 
        x = (node.reshape(mask.shape[0], -1, node.shape[1]) * mask).sum(1) / mask.sum(1)
        return self.head(x)




if __name__ == '__main__':
    num_nodes = 5
    num_edges = 4
    node_features_dim = 3
    edge_features_dim = 2

    # 节点特征 (nodes)
    nodes = torch.rand(1, num_nodes, node_features_dim)  # [num_nodes, node_features_dim]

    # 边索引 (edge_index)
    edge_index = torch.tensor([
        [0, 1, 2, 3],  # 每条边的起点
        [1, 2, 3, 4]   # 每条边的终点
    ], dtype=torch.long)  # [2, num_edges]

    # 边特征 (edge_attr)
    edge_attr = torch.rand(num_edges, edge_features_dim)  # [num_edges, edge_features_dim]

    # 节点的三维位置 (positions)
    positions = torch.rand(num_nodes, 3)  # [num_nodes, 3]

    # 创建 EGNN 模型
    model = EGNN_Network(
        dim=node_features_dim,      # 节点特征维度
        edge_dim=edge_features_dim, # 边特征维度
        num_nearest_neighbors=num_edges,  # 最多的邻居数 
        depth=5
    )

    # 将数据输入到模型中
    updated_nodes, updated_positions = model(nodes, edge_index, edge_attr, positions)
    print(updated_nodes.shape)  # torch.Size([5, 3])
    print(updated_positions.shape)  # torch.Size([5, 3])