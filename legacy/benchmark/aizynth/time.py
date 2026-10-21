import os
import re

def count_words_in_log_files(directory):
    not_count = 0
    is_solved_count = 0
    total_time = 0
    mols = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        match = re.search(r'in (\d+(\.\d+)?) s', line)
                        if match:
        # 提取时长并累加到总时长
                            total_time += float(match.group(1))
                            mols += 1
                            if mols == 10000:
                                print(total_time)
                                exit()
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
models = ['enamine']

for model in models:

    not_count, is_solved_count = count_words_in_log_files("./"+model)

    print(f"在所有 .log 文件中，'not' 出现了 {not_count} 次，'is solved' 出现了 {is_solved_count} 次。")
    print(f"{round(is_solved_count/(is_solved_count+not_count),4)}" )

# 1072 8089