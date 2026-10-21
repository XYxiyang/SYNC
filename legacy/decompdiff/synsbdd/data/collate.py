import torch
import torch.nn.functional as F

from collections.abc import Callable, Mapping, Sequence
from data.preprocessing.smiles2graph import preprocess_item


def collate_tokens(features: Sequence[torch.Tensor], max_size: int) -> torch.Tensor:
    features_padded = [F.pad(f, pad=[0, max_size - f.size(-1)], mode="constant", value=0) for f in features]
    return torch.stack(features_padded, dim=0)

def apply_collate(
    spec: Mapping[str, Callable[[Sequence[torch.Tensor], int], torch.Tensor]],
    data_list: Sequence[dict[str, torch.Tensor]],
    max_size: int,
) -> dict[str, torch.Tensor]:
    transpose = {k: [d[k] for d in data_list] for k in spec.keys()}
    batch = {k: spec[k](transpose[k], max_size) for k in spec.keys()}
    return batch

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

def collator(items, max_node=512, multi_hop_max_dist=20, rel_pos_max=20):


    num_virtual_tokens = 1

    items = [preprocess_item(item) for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            #item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.rel_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.adj,
        )
        for item in items
    ]
    (
        #idxs,
        attn_biases,
        attn_edge_types,
        rel_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        adjs,

    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):

        attn_biases[idx][num_virtual_tokens:, num_virtual_tokens:][
            rel_poses[idx] >= rel_pos_max
        ] = float("-inf")

    # Maximum number of nodes in the batch.
    '''max_node_num = max(i.size(0) for i in xs)
    div = max_node_num // 4
    max_node_num = 4 * div + 3
'''
    max_node_num = max_node
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])

    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [
            pad_attn_bias_unsqueeze(i, max_node_num + num_virtual_tokens)
            for i in attn_biases
        ]
    )
    adj = torch.cat([pad_2d_bool(i, max_node_num + num_virtual_tokens) for i in adjs])
    attn_edge_type = torch.cat(
        [
            pad_edge_type_unsqueeze(i, max_node_num + num_virtual_tokens)
            for i in attn_edge_types
        ]
    )
    rel_pos = torch.cat([pad_rel_pos_unsqueeze(i, max_node_num) for i in rel_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees])


    return Batch(
        #idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        rel_pos=rel_pos,
        in_degree=in_degree,
        out_degree=out_degree,
        x=x,
        edge_input=edge_input,
        y=y,
        adj=adj,
    )


def get_edges_batch(n_nodes, edges, batch_size):

    padded_edge = list(edges)
    if batch_size == 1:
        return edges
    elif batch_size > 1:
        for i in range(batch_size):
            padded_edge[i] += n_nodes * i
    return padded_edge

def collator_3d(items, max_node=512):

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
    mask = x.not_equal(0)

    edge_indices = get_edges_batch(max_node_num, edge_indices, len(ys))
    edge_indices = torch.cat(edge_indices, dim=1)
    #edge_indices = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in edge_indices])
    edge_attrs = torch.cat(edge_attrs).unsqueeze(1).long()
    positions = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in postions]).reshape(-1,  3)

    return Batch3d(
        x=x,
        y=y,
        edge_index=edge_indices,
        edge_attr=edge_attrs,
        positions=positions,
        mask=mask
    )


