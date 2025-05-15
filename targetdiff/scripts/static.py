import glob
import os
import statistics  # Python自带统计模块（3.4+）

def main():
    # 指定要搜索的目录
    path = "./outputs/eval_results"
    # 仅匹配以"vina"开头且后缀".txt"的文件
    pattern = os.path.join(path, "vina*.txt")

    # 分别存放各字段的值
    scores = []
    mins   = []
    docks  = []

    # 遍历符合条件的所有文件
    for filename in glob.glob(pattern):
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                # 假设每行用制表符分割: name \t smiles \t vina_score \t vina_min \t vina_dock
                parts = line.split('\t')
                if len(parts) < 5:
                    continue  # 格式异常就跳过，或者自行处理

                # name   = parts[0]
                # smiles = parts[1]
                try:
                    vina_score = float(parts[2])
                    vina_min   = float(parts[3])
                    vina_dock  = float(parts[4])
                except ValueError:
                    # 如果转换失败，跳过该行或自行处理
                    continue

                scores.append(vina_score)
                mins.append(vina_min)
                docks.append(vina_dock)

    # 如果没有数据行，直接退出
    if len(scores) == 0:
        print("未找到任何有效数据。")
        return

    # 计算平均值和中位数
    mean_score  = statistics.mean(scores)
    median_score= statistics.median(scores)

    mean_min    = statistics.mean(mins)
    median_min  = statistics.median(mins)

    mean_dock   = statistics.mean(docks)
    median_dock = statistics.median(docks)

    # 打印结果
    print(f"=== 统计结果 (共 {len(scores)} 条数据) ===")
    print(f"vina_score: 平均值 = {mean_score:.3f}, 中位数 = {median_score:.3f}")
    print(f"vina_min:   平均值 = {mean_min:.3f},   中位数 = {median_min:.3f}")
    print(f"vina_dock:  平均值 = {mean_dock:.3f},  中位数 = {median_dock:.3f}")

if __name__ == "__main__":
    main()
