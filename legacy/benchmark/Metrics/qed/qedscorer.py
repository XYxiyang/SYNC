from rdkit.Chem.QED import qed
from rdkit import Chem
import time

if __name__ == '__main__':
    models = ["3dsbdd", "d3fg", "decompdiff", "diffbp", "diffsbdd", "flag", "graphbp", "ligan", "molcraft", "pocket2mol", "targetdiff", "voxbind"]
    #models = ["pos", "neg"]
    #models = ["decompdiff"]
    #models = ["gasa_test1_neg", "gasa_test1_pos", "gasa_test2_neg", "gasa_test2_pos", 
    #          "gasa_test3_neg", "gasa_test3_pos", "nonpher_neg", "nonpher_pos"]
    models = ["enamine", "compas"]
    #models = ["guide_12_23"]
    #models = ['decompdiff_guide']
    #models = ['dpo_300']
    #models = ['targetdiff_noedge']
    models = ['targetdiff_dpo_1800']
    models = ['decomp_0509', 'decomp_0510', 'decomp_origin', 'decomp_dpo']

    start_time = time.time()
    for model in models:
        out = open(f"./{model}_qed.csv", "w")
        out.write("smiles,smiles_score\n")
        file = open(f"../smiles/{model}_smiles.txt")
        smiles = file.readlines()
        for i, smile in enumerate(smiles):
            if i% 1000 == 0:
                print(i)
            smile = smile.strip()
            try:
                rdmol = Chem.MolFromSmiles(smile)
                qed_score = qed(rdmol)
                out.write(f'{smile},{qed_score}\n')
            except:
                print(f"Error: {smile}")
            
        file.close()
        out.close()
        
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    


        



