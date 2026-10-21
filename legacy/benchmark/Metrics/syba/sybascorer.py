#from rdkit import Chem
from syba.syba import SybaClassifier

from rdkit.Chem import AllChem as Chem

import time

syba = SybaClassifier()
syba.fitDefaultScore()

def predict_syba(file):
    out = open(f"./{file}_syba.csv", "w")
    #out = open(f"./test_syba.csv", "w")
    out.write("smiles,smiles_score\n")
    file = open(f"../smiles/{file}.txt")
    smiles = file.readlines()
    err = 0
    for smile in smiles:
        smile = smile.strip()
        try:
            syba_score = syba.predict(smile)
            out.write(f'{smile},{syba_score}\n')
        except Exception as e:
            #print(f"{smile} error, {e}")
            err += 1
            #break
    print(f"{file}: {err} errors in total")
    file.close()
    out.close()


    
    

if __name__ == "__main__":
    #models = ["3dsbdd", "d3fg", "decompdiff", "diffbp", "diffsbdd", "flag", "graphbp", "ligan", "molcraft", "pocket2mol", "targetdiff", "voxbind"]
    #for model in models:
    #    predict_syba(model)

    #predict_syba("pos_smiles_50w")
    #predict_syba("neg_smiles_50w")
    #predict_syba("gasa_test1_neg")
    #predict_syba("gasa_test1_pos")
    #predict_syba("gasa_test2_neg")
    #predict_syba("gasa_test2_pos")
    #predict_syba("gasa_test3_neg")
    #predict_syba("gasa_test3_pos")
    #predict_syba("nonpher_neg")
    #predict_syba("nonpher_pos")
    #predict_syba("compas_smiles"
    #predict_syba("enamine_smiles")
    models = ['flowr', 'moldiff']
    models = ['neg_samples']
    models = ['8dyg_origin', '8dyg_dpo']
    for model in models:
        predict_syba(model)
    






