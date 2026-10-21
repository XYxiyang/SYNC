import pandas as pd

models = ["3dsbdd", "d3fg", "decompdiff", "diffbp", "diffsbdd", 
          "flag", "graphbp", "ligan", "molcraft", "pocket2mol", 
          "targetdiff", "voxbind"]

def static_models(model, metric):

    df = pd.read_csv(f'../{metric}/{model}_{metric}.csv')
    mean_value = df['smiles_score'].mean().round(4)  
    variance_value = df['smiles_score'].std().round(3)  

    #print(model)
    print(f"{mean_value} ± {variance_value}")

def static_pair(model, metric):

    df1 = pd.read_csv(f'../{metric}/{model}_pos_{metric}.csv')
    df2 = pd.read_csv(f'../{metric}/{model}_neg_{metric}.csv')

    # 合并两个数据框
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # 计算 'smiles_score' 列的均值和标准差
    mean_score = merged_df['smiles_score'].mean().round(4) 
    std_score = merged_df['smiles_score'].std().round(3)  
    print(f"{mean_score} ± {std_score}")

if __name__ == "__main__":
    #static_models("neg_smiles", "syba")
    #static_models("pos_smiles", "syba")
    #static_models("neg_smiles_50w", "syba")
    #static_models("3dsbdd", "syba")
    #static_models("neg_10w", "ra")
    #static_models("pos_10w", "ra")
    #static_models("neg_50w", "ra")
    #static_models("pos_50w", "ra")
    #static_models("neg", "ra")
    #static_models("pos", "ra")
    #static_models("compas", "ra")
    #static_models("enamine", "ra")
    #static_models("dpo_300", "fs")
    #static_models("dpo_300", "ra")
    #static_models("dpo_300", "sa")
    #static_models("dpo_300", "sc")
    #static_models("dpo_300", "syba")
    #static_models("targetdiff_noedge", 'syba')
    #static_models("targetdiff_noedge", 'ra')
    #static_models("targetdiff_noedge", 'fs')
    #static_models("targetdiff_noedge", 'sc')
    #static_models("targetdiff_noedge", 'sa')
    #static_models("targetdiff_noedge", 'syba')

    #static_models("3dsbdd", 'sa')
    #static_models("d3fg", 'sa')
    #static_models("diffbp", 'sa')
    #static_models("diffsbdd", 'sa')
    #static_models("flag", 'sa')
    #static_models("graphbp", 'sa')
    #static_models("ligan", 'sa')
    #static_models("molcraft", 'sa')
    #static_models("pocket2mol", 'sa')
    #static_models("voxbind", 'sa')
    #static_models("decompdiff", 'sa')
    #static_models("targetdiff", 'sa')
    #static_models("gasa_test1_pos", 'fs')
    #static_models("gasa_test1_neg", 'fs')
    #static_models("gasa_test2_pos", 'fs')
    #static_models("gasa_test2_neg", 'fs')
    #static_models("gasa_test3_pos", 'fs')
    #static_models("gasa_test3_neg", 'fs')
    #static_models("nonpher_pos", 'fs')
    #static_models("nonpher_neg", 'fs')
    #static_models("targetdiff_dpo_1800", "sa")
    #static_models("targetdiff_noedge", "sa")
    #static_models("targetdiff_dpo_1800", "syba")
    #static_models("targetdiff_dpo_1800", "syba")
    #static_models("targetdiff_dpo_1800", "ra")
    #static_models("diffsbdd_caj", "fs")
    #static_models("diffsbdd_caj", "sc")
    
    #static_models("gasa_test1_pos", "qed")
    #static_models("decomp_0510", "qed")
    #static_models("decomp_dpo", "qed")
    #static_pair("gasa_test1", "sc")
    #static_pair("gasa_test2", "sc")
    #static_pair("gasa_test3", "sc")
    #static_pair("nonpher", "sc")
    static_models("8dyg_dpo", "sa")
    static_models("8dyg_dpo", "ra")
    static_models("8dyg_dpo", "syba")
    static_models("8dyg_dpo", "fs")
    static_models("8dyg_dpo", "sc")

#

    
    

    
    
    
    
    


