import os

# 定义根目录路径
root_dir = './outputs_guide'  # 替换为你的目录

# 遍历根目录下的所有子目录
for item in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, item)
    if os.path.isdir(subdir_path):
        # 检查子目录中是否存在 "result.pt" 文件
        result_file = os.path.join(subdir_path, "result.pt")
        if not os.path.exists(result_file):
            print(f"{subdir_path} 没有 result.ptz 文件")
