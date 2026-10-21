from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDWriter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import itertools
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
import time

# 以 SMILES 为例
def read_smiles_from_txt(file_path):
    smiles_list = []
    with open(file_path, 'r') as file:
        for line in file:
            smiles = line.strip()
            if smiles:  
                smiles_list.append(smiles)
    return smiles_list

def process_smiles_list(smiles_list, output_sdf_file, num_processes=None):
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)

    print(num_processes)
    with Pool(processes=num_processes) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(process_molecule, smiles_list), total=len(smiles_list)):
            results.append(result)

    valid_molecules = [mol for mol in results if mol is not None]
    print(len(valid_molecules))
    with SDWriter(output_sdf_file) as writer:
        for mol in valid_molecules:
            for conf in mol.GetConformers():
        # 写入每个构象到同一个 SDF 文件中，使用 conf.GetId() 作为构象的 id
                writer.write(mol, confId=conf.GetId())


# 使用 ETKDG 方法生成 10 个构象
def process_molecule(smile):
    try:
        m = Chem.MolFromSmiles(smile)
        if m is None:
            return None 
        m = Chem.AddHs(m) 
        params = AllChem.ETKDG()
        conformer_ids = AllChem.EmbedMultipleConfs(m, numConfs=1, params=params)

        # 对每个构象进行几何优化（可选）
        for conf_id in conformer_ids:
            AllChem.UFFOptimizeMolecule(m, confId=conf_id)
        return m
    except RuntimeError as e:
        print(f"Failed to generate conformer for SMILES: {smile}. Error: {e}")
    

if __name__ == "__main__":
    
    #smiles = read_smiles_from_txt("./gasa_neg.txt")
    #process_smiles_list(smiles, "gasa_neg_5.sdf")
    #smiles = read_smiles_from_txt("./gasa_pos.txt")
    #process_smiles_list(smiles, "gasa_pos_5.sdf")
    #smiles = read_smiles_from_txt("nonpher_neg.txt")
    #process_smiles_list(smiles, "nonpher_neg_5.sdf")

    start = time.time()
    smiles = read_smiles_from_txt("1000.txt")
    process_smiles_list(smiles, "1000.sdf")
    print(time.time() - start)
    