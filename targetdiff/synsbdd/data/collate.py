import torch
import torch.nn.functional as F
import pdb

from collections.abc import Callable, Mapping, Sequence

from sklearn.neighbors import NearestNeighbors
import copy



def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_bool(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(False)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def collate_graph(batch):

    max_atoms_len = max(len(sample['atoms']) for sample in batch)
    
    atoms_list = []
    bonds_list = []
    y_list = []
    atom_masks_list = []
    
    for sample in batch:
        atoms = sample['atoms']
        bonds = sample['bonds']
        y = sample['y']
        
        padded_atoms = F.pad(atoms, (0, max_atoms_len - len(atoms)), value=0)  
        atoms_list.append(padded_atoms)
        

        atom_mask = torch.ones(max_atoms_len, dtype=torch.bool)  # 默认全是有效
        atom_mask[len(atoms):] = 0  # 填充部分为 0
        atom_masks_list.append(atom_mask)
        
        # 4. Padding bonds：填充到最大 size
        padded_bonds = F.pad(bonds, (0, max_atoms_len - bonds.size(1), 0, max_atoms_len - bonds.size(0)), value=0)
        bonds_list.append(padded_bonds)
        
        y_list.append(y)
    
    # 5. 堆叠 atoms, bonds, y 和 mask
    atoms_tensor = torch.stack(atoms_list, dim=0)  # [batch_size, max_atoms_len]
    bonds_tensor = torch.stack(bonds_list, dim=0)  # [batch_size, max_atoms_len, max_atoms_len]
    y_tensor = torch.stack(y_list, dim=0)          # [batch_size, 1]
    atom_masks = torch.stack(atom_masks_list, dim=0)  # [batch_size, max_atoms_len]
    
    return {'atoms': atoms_tensor, 'bonds': bonds_tensor, 'atom_padding_mask': atom_masks, 'y': y_tensor}
class Batch:
    def __init__(
        self,
        #idx,
        attn_bias,
        attn_edge_type,
        rel_pos,
        in_degree,
        out_degree,
        x,
        edge_input,
        y,
        adj,
    ):
        super(Batch, self).__init__()
        #self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.attn_bias, self.attn_edge_type, self.rel_pos = (
            attn_bias,
            attn_edge_type,
            rel_pos,
        )
        self.edge_input = edge_input
        self.adj = adj

    def to(self, device):
        #self.idx = self.idx.to(device)
        self.in_degree, self.out_degree = (
            self.in_degree.to(device),
            self.out_degree.to(device),
        )
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias, self.attn_edge_type, self.rel_pos = (
            self.attn_bias.to(device),
            self.attn_edge_type.to(device),
            self.rel_pos.to(device),
        )
        self.edge_input = self.edge_input.to(device)
        self.adj = self.adj.to(device)
        return self

    def __len__(self):
        return self.in_degree.size(0)
    
class Batch3d:
    def __init__(
        self, x, y, edge_index, edge_attr, positions, mask
    ):
        super(Batch3d, self).__init__()
        #self.idx = idx
        self.x, self.y = x, y
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.positions = positions
        self.mask = mask

    def to(self, device):
        #self.idx = self.idx.to(device)
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.positions = self.positions.to(device)
        self.mask = self.mask.to(device)
        return self

    def __len__(self):
        return self.y.size(0)
    
class Batch3d_dense:
    def __init__(
        self, x, y, adj_matrice, positions, node_mask, adj_mask
    ):
        super(Batch3d_dense, self).__init__()
        #self.idx = idx
        self.x, self.y = x, y
        self.adj_matrice = adj_matrice
        self.positions = positions
        self.node_mask = node_mask
        self.adj_mask = adj_mask

    def to(self, device):
        #self.idx = self.idx.to(device)
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.adj_matrice = self.adj_matrice.to(device)
        self.positions = self.positions.to(device)
        self.node_mask = self.node_mask.to(device)
        self.adj_mask = self.adj_mask.to(device)
        return self

    def __len__(self):
        return self.y.size(0)


def get_edges_batch(n_nodes, edges, batch_size):

    padded_edge = list(edges)
    if batch_size == 1:
        return edges
    elif batch_size > 1:
        for i in range(batch_size):
            padded_edge[i] += n_nodes * i
    return padded_edge

def collator_3d_sbdd(items, max_node=512):

    items = [
        (
            item.x,
            item.y,
            item.edge_index,
            item.edge_attr,
            item.positions
        )
        for item in items
    ]
    (
        xs,
        ys,
        edge_indices,
        edge_attrs,
        postions,
    ) = zip(*items)


    max_node_num = max_node

    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs]).float().reshape(-1,  158) #originally .long()
    mask = (x[:, 0] > 0).view(-1, max_node, 1).float()

    edge_indices = get_edges_batch(max_node_num, edge_indices, len(ys))
    edge_indices = torch.cat(edge_indices, dim=1)
    #edge_indices = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in edge_indices])
    edge_attrs = torch.cat(edge_attrs).float() #originally unsqueeze(1)
    positions = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in postions]).reshape(-1,  3)

    return Batch3d(
        x=x, # [b*n, d(1;158)]
        y=y, # [b, 1]
        edge_index=edge_indices, #[2, e]
        edge_attr=edge_attrs, # [e, d(1)]
        positions=positions, # [b*n, 3]
        mask=mask # [b, n, 1]
    )


def collator_3d_lin(items, max_node=512):

    items = [
        (
            item.x,
            item.y,
            item.edge_index,
            item.edge_attr,
            item.positions
        )
        for item in items
    ]
    (
        xs,
        ys,
        edge_indices,
        edge_attrs,
        postions,
    ) = zip(*items)


    max_node_num = max_node

    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    mask = x.not_equal(0).sum(dim=-1).unsqueeze(-1).float()
    x = x.reshape(-1, x.shape[-1]).float()
    

    edge_indices = get_edges_batch(max_node_num, edge_indices, len(ys))
    edge_indices = torch.cat(edge_indices, dim=1)
    #edge_indices = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in edge_indices])
    edge_attrs = torch.cat(edge_attrs).float()
    positions = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in postions]).reshape(-1,  3)

    return Batch3d(
        x=x, #[b*n, d]
        y=y, 
        edge_index=edge_indices, #[2, e]
        edge_attr=edge_attrs, # [e, d]
        positions=positions, # [b*n, 3]
        mask=mask
    )


def collator_3d_rebuild(items, max_node=512):

    items = [
        (
            item.x,
            item.y,
            item.edge_index,
            item.edge_attr,
            item.positions
        )
        for item in items
    ]
    (
        xs,
        ys,
        edge_indices,
        edge_attrs,
        postions,
    ) = zip(*items)


    max_node_num = max_node
    num_graphs = len(ys)

    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(F.one_hot(i, 158), max_node_num) for i in xs], dim=0).float()
    mask = x.not_equal(0).sum(dim=-1).unsqueeze(-1).float()
    #x = x.reshape(-1, x.shape[-1]).float()

    adj_matrice = torch.zeros((num_graphs, max_node_num, max_node_num))
    adj_mask = torch.zeros((num_graphs, max_node_num, max_node_num), dtype=bool)
    for n in xs:
        adj_mask[:len(n), :len(n)] = True 

    
    for i in range(num_graphs):
        for j in range(edge_indices[i].size(1)):
            src, dst = edge_indices[i][:, j]
            adj_matrice[i][src][dst] = edge_attrs[i][j]
    

    #edge_indices = get_edges_batch(max_node_num, edge_indices, len(ys))
    #edge_indices = torch.cat(edge_indices, dim=1)
    ##edge_indices = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in edge_indices])
    #edge_attrs = torch.cat(edge_attrs).float()
    positions = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in postions])

    return Batch3d_dense(
        x=x, #[b, n, d]
        y=y, 
        #edge_index=edge_indices, #[2, e]
        #edge_attr=edge_attrs, # [e, d]
        adj_matrice=adj_matrice, #[b, n, n]
        positions=positions, # [b, n, 3]
        node_mask=mask, #[b, n ,1]
        adj_mask=adj_mask #[b, n, n]
    )


def collator_3d_emb(items, max_node=512):

    items = [
        (
            item.x.unsqueeze(1),
            item.y,
            item.edge_index,
            item.edge_attr,
            item.positions
        )
        for item in items
    ]
    (
        xs,
        ys,
        edge_indices,
        edge_attrs,
        postions,
    ) = zip(*items)


    max_node_num = max_node

    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs]).long()
    mask = x.not_equal(0)
    

    edge_indices = get_edges_batch(max_node_num, edge_indices, len(ys))
    edge_indices = torch.cat(edge_indices, dim=1)
    #edge_indices = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in edge_indices])
    edge_attrs = torch.cat(edge_attrs).long()
    positions = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in postions]).reshape(-1,  3)

    return Batch3d(
        x=x, #[b*n, d]
        y=y, 
        edge_index=edge_indices, #[2, e]
        edge_attr=edge_attrs, # [e, d]
        positions=positions, # [b*n, 3]
        mask=mask
    )

def collator_3d_knn(items, max_node=512):

    items = [
        (
            item.x.unsqueeze(1),
            item.y,
            item.edge_index,
            item.edge_attr,
            item.positions
        )
        for item in items
    ]
    (
        xs,
        ys,
        edge_indices,
        edge_attrs,
        postions,
    ) = zip(*items)


    max_node_num = max_node

    y = torch.cat(ys)
    

    edges = []
    for pos in postions:
        edge_start = []
        edge_end = []
        knn = NearestNeighbors(n_neighbors=4)  # 找每个点的 3 个最近邻
        knn.fit(pos.numpy())  # 注意：scikit-learn 要使用 numpy 数据
        distances, indices = knn.kneighbors(pos.numpy())
        for i in range(len(pos)):
            for j in range(1, len(indices[i])):  # 排除自己，索引从 1 开始
                if distances[i][j] < 2.0:
                    edge_start.append(i)
                    edge_end.append(indices[i][j])
        edges.append(torch.tensor([edge_start, edge_end], dtype=torch.long))
    
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs]).long()
    mask = x.not_equal(0)
    

    edge_indices = get_edges_batch(max_node_num, edges, len(ys))
    edge_indices = torch.cat(edge_indices, dim=1)
    #edge_indices = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in edge_indices])
    edge_attrs = torch.cat(edge_attrs).long()
    positions = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in postions]).reshape(-1,  3)

    return Batch3d(
        x=x, #[b*n, d]
        y=y, 
        edge_index=edge_indices, #[2, e]
        edge_attr=edge_attrs, # [e, d]
        positions=positions, # [b*n, 3]
        mask=mask
    )



def collator_3d_edge(items, max_node=512):

    items = [
        (
            item.x.unsqueeze(1),
            item.y,
            item.edge_index,
            item.edge_attr,
            item.positions
        )
        for item in items
    ]
    (
        xs,
        ys,
        edge_indices,
        edge_attrs,
        postions,
    ) = zip(*items)


    max_node_num = max_node
    bsz = len(ys)

    y = torch.cat(ys)
    
    

    edges = []
    for pos in postions:
        edge_start = []
        edge_end = []
        knn = NearestNeighbors(n_neighbors=5)  # 找每个点的 3 个最近邻
        knn.fit(pos.numpy())  # 注意：scikit-learn 要使用 numpy 数据
        distances, indices = knn.kneighbors(pos.numpy())
        for i in range(len(pos)):
            for j in range(1, len(indices[i])):  # 排除自己，索引从 1 开始
                #if distances[i][j] < 2.0:
                edge_start.append(i)
                edge_end.append(indices[i][j])
        edges.append(torch.tensor([edge_start, edge_end], dtype=torch.long))
    
    #x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs]).long()
    #mask = x.not_equal(0)

    n_nodes = [0]
    for x in xs:
        n_nodes.append(x.size(0))

    

    knn_egdes = list(edges)
    for i in range(bsz):
        knn_egdes[i] += n_nodes[i] # [b, 2, 4n]
    


    origin_edges = list(edge_indices) # [b, 2, e]
    for i in range(bsz):
        #origin_edges[i] = sorted(origin_edges[i], key=lambda x: x[0]) #ascending start node order
        origin_edges[i] += n_nodes[i]

    padded_edge_attr = []
    
    for i in range(bsz):
        attrs = []
        #print(xs[i].size(0) * 4)
        #print(knn_egdes[i].size(1))
        if xs[i].size(0) * 4 != knn_egdes[i].size(1):
            print("error")
        #assert xs[i].size(0) * 4 == knn_egdes[i].size(1), pdb.set_trace()
        for idx in range(xs[i].size(0) * 4): #4n edges
            if idx >= knn_egdes[i].size(1):
                continue
           
            start = knn_egdes[i][0][idx]
            end = knn_egdes[i][1][idx]
            edge_exists = ((origin_edges[i][0] == start) & (origin_edges[i][1] == end))
            if edge_exists.any():
                edge_indices = torch.nonzero(edge_exists).squeeze()
                attrs.append(edge_attrs[i][edge_indices])
            else:
                attrs.append(0)
        padded_edge_attr.append(torch.tensor(attrs))
            #if edge_exists:
            #    attrs.append(edge_attrs[i][idx])
#
            #if node not in origin_edges[i]:
            #    origin_edges[i].append(node)
            #    origin_edges[i].append(node)
#
    #padded_edge_attr = 
    knn_egdes = torch.cat(knn_egdes, dim=1)
    #origin_edges = torch.cat(origin_edges, dim=1)
    x = torch.cat(xs)

    #edge_indices = get_edges_batch(max_node_num, edges, len(ys))
    #edge_indices = torch.cat(edge_indices, dim=1)
    #edge_indices = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in edge_indices])
    padded_edge_attr = torch.cat(padded_edge_attr).long()
    positions = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in postions]).reshape(-1,  3)

    return Batch3d(
        x=x, #[n, d]
        y=y, 
        edge_index=knn_egdes,
        #origin_edges=origin_edges, #[2, e]
        edge_attr=padded_edge_attr, # [e, d]
        positions=positions, # [b*n, 3]
        mask=torch.ones(1) #simply occupy a place
    )

