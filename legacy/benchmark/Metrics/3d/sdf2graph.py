from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
from tqdm import tqdm

def sdf_to_graph(sdf_file):
    # 加载 SDF 文件
    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
    graphs = []

    for mol in tqdm(supplier, desc="SDF to graph", total=len(supplier)):
        if mol is None:  # 跳过无效分子
            continue

        # 1. 提取节点（atoms）特征
        atoms = []
        for atom in mol.GetAtoms():
            atoms.append(atom.GetAtomicNum())  # 使用原子序数作为特征
        atoms = torch.tensor(atoms, dtype=torch.long)  # 转换为张量

        # 2. 提取边（edge_index 和 edge_attr）特征
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()

            # 双向边
            edge_index.append([start, end])
            edge_index.append([end, start])

            # 边特征：键类型
            bond_feature = [
                1 if bond_type == Chem.rdchem.BondType.SINGLE else 0,
                1 if bond_type == Chem.rdchem.BondType.DOUBLE else 0,
                1 if bond_type == Chem.rdchem.BondType.TRIPLE else 0,
                1 if bond_type == Chem.rdchem.BondType.AROMATIC else 0,
            ]
            edge_attr.append(bond_feature)
            edge_attr.append(bond_feature)  # 双向边需要特征重复

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 转置为 [2, num_edges]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # 3. 提取原子坐标（positions）
        conformer = mol.GetConformer()
        positions = []
        for i in range(mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])
        positions = torch.tensor(positions, dtype=torch.float)  # 转换为张量

        # 4. 构建图数据结构
        graph = {
            "atoms": atoms,             # 节点特征
            "edge_index": edge_index,   # 边索引
            "edge_attr": edge_attr,     # 边特征
            "positions": positions      # 节点 3D 坐标
        }
        graphs.append(graph)

    return graphs

# 示例用法
sdf_file = "output.sdf"  # 替换为你的 SDF 文件路径
graphs = sdf_to_graph(sdf_file)
print(len(graphs))

#torch.save(graphs, "80_pos.pt")  # 保存图数据

# 打印第一个分子的图信息
for i, graph in enumerate(graphs):
    print(f"Graph {i+1}:")
    print("Atoms:", graph['atoms'].shape)
    print("Edge Index:", graph['edge_index'].shape)
    print("Edge Attr:", graph['edge_attr'])
    print("Positions:", graph['positions'].shape)
