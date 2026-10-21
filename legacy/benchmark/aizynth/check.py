import os

def count_words_in_log_files(directory):
    not_count = 0
    is_solved_count = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        not_count += line.count('not')
                        is_solved_count += line.count('is solved')
                        #if 'is solved' in line:
                        #    print(line.strip())

    return not_count, is_solved_count

directory_path = './pos_10w'  

models = [ "d3fg", "decompdiff", "diffbp", "diffsbdd", 
          "flag", "graphbp", "ligan", "molcraft", "pocket2mol", 
          "targetdiff", "voxbind"]
models = ['decompdiff_300']
models = ['gasa3neg', 'gasa3pos']
models = ["guide_0"]
models = ["pocket1_origin", "guide_12_23", 'guide_12_22']
models = ["dpo_500", "dpo_300"]
models = ["decompdiff_22_7"]
models = ["diffsbdd_caj"]
models = ['targetdiff_dpo_1800']
models = ['9fl3_dpo']
models = ['targetdiff_noedge']
models = ['pkt2molall', 'projected']
models = ['decomp_0510', 'decomp_0509', 'decomp_origin', 'decomp_dpo_1']
models = ['gasa1neg', 'gasa1pos', 'gasa2neg', 'gasa2pos','gasa3neg', 'gasa3pos', 'nonpher_neg']

models = ['flowr', 'moldiff']
models = ['decomp_origin']
models = ['new_tgt/8DYG_origin', 'new_tgt/8DYG_guide', 'new_tgt/8DYG_DPO']
models = ['8dyg', '8dyg_dpo']
models = ['9fl3', '9fl3_dpo']
for model in models:

    not_count, is_solved_count = count_words_in_log_files("./"+model)

    print(f"在所有 .log 文件中，'not' 出现了 {not_count} 次，'is solved' 出现了 {is_solved_count} 次。")
    print(f"{round(is_solved_count/(is_solved_count+not_count),4)}" )

# 1072 8089