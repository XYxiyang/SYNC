import pandas as pd
import os
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDWriter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

def process_molecule(smile):
    try:
        m = Chem.MolFromSmiles(smile)
        if m is None:
            return None 
        m = Chem.AddHs(m) 
        if AllChem.EmbedMolecule(m, randomSeed=1) != 0:
            return None
        #AllChem.UFFOptimizeMolecule(m)
        return m
    except RuntimeError as e:
        print(f"Failed to generate conformer for SMILES: {smile}. Error: {e}")

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

    with Pool(processes=num_processes) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(process_molecule, smiles_list), total=len(smiles_list)):
            results.append(result)

    #results = []
    #for smile in tqdm(smiles_list, total=len(smiles_list)):
    #    #print(smile)
    #    result = process_molecule(smile)  # 单线程处理
    #    results.append(result)

    valid_molecules = [mol for mol in results if mol is not None]
    print(len(valid_molecules))
    with SDWriter(output_sdf_file) as writer:
        for mol in valid_molecules:
            writer.write(mol)
    return valid_molecules

def gasatest(idx :int):
    df = pd.read_csv(f'data/smiles/gasa_test{idx}.csv')
    os.makedirs("data/processed/sdf/", exist_ok=True)

    #filtered_df = df[df['labels'] == 1]
    #process_smiles_list(filtered_df['smiles'].tolist(), f"data/processed/sdf/gasa_test{idx}_neg.sdf")
    filtered_df = df[df['labels'] == 0]
    process_smiles_list(filtered_df['smiles'].tolist(), f"data/processed/sdf/gasa_test{idx}_pos.sdf")

def nonpher_test():
    df = pd.read_csv(f'data/smiles/nonpher_test.csv')
    os.makedirs("data/processed/sdf/", exist_ok=True)

    filtered_df = df[df['synthetic_feasibility'] == "infeasible"]
    process_smiles_list(filtered_df['smiles'].tolist(), f"data/processed/sdf/nonpher_test_neg.sdf")
    filtered_df = df[df['synthetic_feasibility'] == "feasible"]
    process_smiles_list(filtered_df['smiles'].tolist(), f"data/processed/sdf/nonpher_test_pos.sdf")

def sbdd(name: str):
    sbdd_smiles = read_smiles_from_txt(f"smiles/{name}_smiles.txt")
    os.makedirs("sbdd/sdf/", exist_ok=True)
    process_smiles_list(sbdd_smiles, f'sbdd/sdf/{name}.sdf')

if __name__ == "__main__":

    #smiles = read_smiles_from_txt("/data/smiles/gasa_test1.txt")
    #smiles = ["CCO", "CCC"]
    #process_smiles_list(smiles, '80_neg.sdf')
    #writer.close()
    #gasatest(1)
    #gasatest(2)
    #gasatest(3)
    #nonpher_test()
    models = ["3dsbdd", "d3fg", "decompdiff", "diffbp", "diffsbdd", "flag", 
              "graphbp", "ligan", "molcraft", "pocket2mol", "targetdiff", "voxbind"]
    
    models = ["guide_1_10", "guide_1_50", "guide_1_100", "guide_1_200", "guide_1_500", "guide_1_1000"]

    models = ["guide_1_aro"]
    models = ['targetdiff_guide_1', "targetdiff_regen_1"]
    models = ["pocket_1"]
    
    for model in models:
        sbdd(model)
