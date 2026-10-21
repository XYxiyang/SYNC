
from RAscore import RAscore_XGB #For XGB based models
import time

xgb_scorer = RAscore_XGB.RAScorerXGB()

#Imatinib mesylate

models = ["3dsbdd", "d3fg", "decompdiff", "diffbp", "diffsbdd", "flag", "graphbp", "ligan", "molcraft", "pocket2mol", "targetdiff", "voxbind"]
models = ["flowr", "moldiff"]

models = ['neg_samples']
models = ['8dyg_origin', '8dyg_dpo']

def predict_ra(model):
    out = open(f"./{model}_ra.csv", "w")
    out.write("smiles,smiles_score\n")
    file = open(f"../Metrics/smiles/{model}.txt")
    smiles = file.readlines()
    err = 0
    for i,smile in enumerate(smiles):
        if(i%1000==0):
            print(i)
        smile = smile.strip()
        try:
            rs_score = xgb_scorer.predict(smile)
            out.write(f'{smile},{rs_score}\n')
        except Exception as e:
            #print(f"{smile} error, {e}")
            err += 1
    print(f"{model}: {err} errors in total")
    file.close()
    out.close()

if __name__ == "__main__":
    #predict_ra("compas_smiles")
    #predict_ra("enamine_smiles")
    
    #predict_ra("neg_smiles")
    #predict_ra("dpo_700_smiles")

    '''predict_ra("flowr")
    predict_ra("neg_samples")'''
    predict_ra("8dyg_origin")
    predict_ra("8dyg_dpo")


