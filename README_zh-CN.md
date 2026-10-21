# SYNC — ICLR 2026

[English](README.md) | [简体中文](README_zh-CN.md)

论文 **SYNC: Measuring and Advancing Synthesizability in Structure-Based Drug Design** 的官方实现。

[论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/ffdb280e7c7b4c4af30e04daf5a84b98-Paper-Conference.pdf)
· [OpenReview](https://openreview.net/forum?id=y1tPw4Uuzg)
· [GitHub](https://github.com/XYxiyang/SYNC)

SYNC 利用分子的三维结构评估可合成性，并通过分类器指导和偏好优化，提升基于结构的分子生成能力。本发布包提供预训练模型、SDF 分子评分、分类器训练、TargetDiff/DecompDiff 指导与 DPO，以及基准评估工具。

以下命令均在本 README 所在目录执行。

## 预训练分类器

包内分类器复现了论文 Table 1 中的以下准确率：

| TS1 | TS2 | TS3 | Nonpher-Test | Enamine |
|---:|---:|---:|---:|---:|
| 0.9911 | 0.8406 | 0.7564 | 0.9313 | 0.9520 |

评估样本数、数值核验和软件测试结果见[验证说明](docs/VALIDATION.md)。

## 三维分子评分

独立评分接口依赖 PyTorch 和 RDKit，支持 CPU 与 CUDA。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .

sync-score --input examples/molecules.sdf \
  --checkpoint checkpoints/sync_noedge.pt \
  --output outputs/example_scores.csv
```

`--input` 支持包含多个分子的 SDF 文件，也支持递归读取目录中的 SDF 文件。安装 CUDA 版 PyTorch 后，可添加 `--device cuda:0` 使用 GPU。

评分时使用输入分子的三维坐标与化学键，移除氢原子，并通过 SYNC 词表编码芳香原子。

输出 CSV 包含：

- `probability_es`：模型预测分子属于易合成类别的概率。
- `prediction`：分类结果，难合成 HS = 0，易合成 ES = 1。
- 分子标识、SMILES 和每条记录的处理状态。

同名 JSON 汇总文件记录输入数量、成功与失败数量，以及被判定为易合成分子的比例。

## 分子生成

在 TargetDiff 采样过程中，SYNC 指导同时更新原子类型概率和坐标。包内提供张量格式的 1WN6 蛋白口袋及两个分子的生成示例。GPU 环境安装步骤见[生成指南](docs/GENERATION.md)。

```bash
python scripts/sample_targetdiff.py \
  --pocket examples/1wn6_pocket.pt --num-samples 2 --batch-size 2 \
  --seed 2021 --output outputs/1wn6 --device cuda:0
```

输出包括生成分子的 SDF、原子类型、坐标、扩散轨迹和 JSON 汇总文件。生成指南还介绍了两套生成模型的完整 CrossDocked 采样、偏好样本对构建及 DPO 流程。

## 训练与评估

```bash
python scripts/prepare_classifier_data.py \
  --manifest /path/to/labeled_sdfs.csv --output data/classifier
python scripts/train_classifier.py \
  --train data/classifier/train.pt --val data/classifier/val.pt \
  --device cuda:0 --output outputs/classifier
python scripts/evaluate_classifier.py \
  --graphs data/classifier/test.pt --checkpoint outputs/classifier/best.pt \
  --output outputs/classifier/test.json
```

标签约定、分子分组、输入格式和基准数据转换见[数据说明](docs/DATA.md)。基于分子指纹、SMILES、二维图及边特征的分类器变体位于 `legacy/classifier/`。

## 目录结构

| 路径 | 内容 |
|---|---|
| `sync_sbdd/` | 独立 EGNN 分类器与 SDF 评分接口 |
| `checkpoints/` | 张量格式的主 SYNC 模型权重 |
| `scripts/` | 数据准备、训练、评估与采样命令 |
| `legacy/classifier/` | 分类器变体与预处理模块 |
| `legacy/targetdiff/` | TargetDiff 指导、DPO、配置与权重 |
| `legacy/decompdiff/` | DecompDiff 指导、DPO、配置与权重 |
| `legacy/benchmark/` | 指标计算与统计分析工具 |
| `examples/` | 示例分子与 1WN6 蛋白口袋 |
| `docs/` | 使用方法、数据说明、验证结果与来源记录 |
| `tests/` | 数值等价性、不变性、梯度与接口测试 |

训练数据集、CrossDocked 数据和偏好轨迹按照[数据说明](docs/DATA.md)与[生成指南](docs/GENERATION.md)中的路径单独配置。指标文件格式与分类阈值见[基准评估说明](docs/BENCHMARKS.md)。

## 检查与来源

```bash
python -m pip install -e ".[validation]"
python -m pytest -q
python scripts/verify_release.py
python scripts/check_assets.py targetdiff
python scripts/check_assets.py decompdiff
```

资产检查工具会报告配置中的模型和数据集路径。`docs/source_manifest.json` 记录源文件及其哈希值，`MANIFEST.sha256` 覆盖发布包中的文件。

第三方组件的来源与许可证见[组件说明](THIRD_PARTY.md)。

## 引用

```bibtex
@inproceedings{liu2026sync,
  title={SYNC: Measuring and Advancing Synthesizability in Structure-Based Drug Design},
  author={Liu, Yunfan and Wu, Lirong and Gao, Zhifeng and Huang, Yufei and
          Tan, Cheng and Lin, Haitao and Liu, Zicheng and Chi, Changxi and
          Yu, Chang and Li, Stan Z.},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=y1tPw4Uuzg}
}
```
