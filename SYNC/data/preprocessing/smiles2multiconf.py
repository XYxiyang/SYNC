from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm  
from utils.misc import read_smiles_from_txt

# 示例分子列表（SMILES 格式）
smiles_list = read_smiles_from_txt("data/smiles/gasa_pos.txt")
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

# 添加氢原子，并确保分子有效
for i, mol in enumerate(mols):
    if mol is not None:
        mols[i] = Chem.AddHs(mol)

# 初始化 SDF 文件写入
output_file = "data/processed/sdf/80_neg_aug.sdf"
writer = Chem.SDWriter(output_file)

# 设置构象生成参数
num_confs = 5  # 每个分子生成的构象数量
params = AllChem.ETKDGv3()  # 使用 ETKDG 算法

# 遍历分子生成构象
for mol_id, mol in enumerate(tqdm(mols, desc="Processing molecules", unit="molecule")):
    if mol is None:
        continue  # 跳过无效分子
    

    # 生成构象
    conformer_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)

    # 优化每个构象
    for conf_id in conformer_ids:
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

    # 为每个构象添加元数据，并写入 SDF 文件
    for conf_id in conformer_ids:
        mol.SetProp("_Name", f"Molecule_{mol_id+1}_Conformer_{conf_id}")
        writer.write(mol, confId=conf_id)

# 关闭文件写入
writer.close()
print(f"所有分子的构象已保存至 {output_file}")
