import pandas as pd

models = ["3dsbdd", "d3fg", "decompdiff", "diffbp", "diffsbdd", 
          "flag", "graphbp", "ligan", "molcraft", "pocket2mol", 
          "targetdiff", "voxbind"]

def static_models(model, metric, threshold, max_is_good=True):
    df = pd.read_csv(f'../{metric}/{model}_pos_{metric}.csv')
    # 筛选出 smiles_score > threshold 的数据
    total = 0
    total += len(df)
    correct = 0

    df_filtered = df[df['smiles_score'] > threshold]
    correct += len(df_filtered)

    df = pd.read_csv(f'../{metric}/{model}_neg_{metric}.csv')
    total += len(df)
    df_filtered = df[df['smiles_score'] < threshold]
    correct += len(df_filtered)
    
    if max_is_good:
        print(f"{metric} Percentage of correct > {correct/total:.4f}")
    else:
        print(f"{metric} Percentage of correct > {(total-correct)/total:.4f}")

def test(metric, threshold, max_is_good=True):
    #static_models("gasa_test1",metric,threshold,max_is_good)
    #static_models("gasa_test2",metric,threshold,max_is_good)
    #static_models("gasa_test3",metric,threshold,max_is_good)
    #static_models("nonpher",metric, threshold,max_is_good)
    static_models("compas_filter","sc", 3.7,False)
    static_models("coconut_5w","sc", 3.7,False)

def enamine_compas(metric, threshold, max_is_good=True):
    df = pd.read_csv(f'../{metric}/coconut_5w_{metric}.csv')
    # 筛选出 smiles_score > threshold 的数据
    total = 0
    total += len(df)
    correct = 0

    df_filtered = df[df['smiles_score'] > threshold]
    correct += len(df_filtered)

    if max_is_good:
        print(f"{metric} Percentage of correct > {correct/total:.4f}")
    else:
        print(f"{metric} Percentage of correct > {(total-correct)/total:.4f}")

    

    df = pd.read_csv(f'../{metric}/compas_filter_{metric}.csv')
    # 筛选出 smiles_score > threshold 的数据
    total = 0
    correct = 0

    total += len(df)
    df_filtered = df[df['smiles_score'] < threshold]
    correct += len(df_filtered)
    
    if max_is_good:
        print(f"{metric} Percentage of correct > {correct/total:.4f}")
    else:
        print(f"{metric} Percentage of correct > {(total-correct)/total:.4f}")

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
    #static_models("targetdiff_noedge", 'qed')

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

    
    #test("fs", -13)
    ##test("ra", 0.5)
    #test("sa", 0.61)
    ##test("syba", -18.6)
    #test("qed", 0.6)
    #test("sc", 3.1,False)

    enamine_compas("fs", -13)
    enamine_compas("ra", 0.5)
    enamine_compas("sa", 0.61)
    enamine_compas("syba", -18.6)
    enamine_compas("qed", 0.6)
    enamine_compas("sc", 3.1,False)
    #test("qed", 3.7)
    #static_models("compas","sc", 3.7,False)
    #static_models("enamine","sc", 3.7,False)

    
    
    
    
    


