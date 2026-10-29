# Spec-o3 快速开始

按顺序完成：安装环境 → 下载数据 → SFT 冷启动 → GRPO → 评测。

原理与实验结果见 [Spec-o3 Blog](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/spec-o3/readme.md)，公开训练曲线见 [SwanLab](https://swanlab.cn/@kmno4/agentic-rl-lab-spec-o3/v1/rjj3zx/runs)。

## 1. 获取项目、安装环境并登录

先安装好 UV，再执行：

```bash
git clone https://github.com/KMnO4-zx/agentic-rl-lab.git
cd agentic-rl-lab/experiments/spec-o3
uv sync --locked

uv run trio login
uv run swanlab login
```

**后续命令都在 `experiments/spec-o3/` 中执行。** 已有仓库时，直接进入这个目录，从 `uv sync --locked` 开始即可。

项目要求 Python 3.13+，锁定 PyTRIO 0.2.9。本地负责数据、工具绘图和任务调度，模型训练与采样由 PyTRIO 远程执行。SwanLab 默认记录到云端；训练命令改用 `--swanlab-mode disabled` 时，可跳过 SwanLab 登录。

## 2. 下载并整理数据

```bash
uv run python prepare_data.py sft
uv run python prepare_data.py bench
```

先完成 `sft`，再执行 `bench`：后者需要读取 SFT 的对象列表来划分开发集。数据与图片直接保存在项目内：

| 路径 | 用途 |
| --- | --- |
| `datasets/sft/sft_train.jsonl` | SFT 训练，848 条 |
| `datasets/sft/sft_val.jsonl` | SFT 验证，94 条 |
| `datasets/sft/images/` | SFT 轨迹图片 |
| `datasets/rl/bench_rl_train.jsonl` | RL 训练，3,108 条 |
| `datasets/rl/bench_dev.jsonl` | 开发集评测，256 条 |
| `datasets/rl/bench_test.jsonl` | 完整测试集，6,754 条 |
| `datasets/rl/images/` | 初始光谱图 |
| `datasets/rl/spectra/` | 工具绘图所需的光谱数组 |

重新执行会复用已下载文件。准备结果中的图片与光谱路径为绝对路径，移动目录或换机器后需重新整理。文本清理记录保存在 `datasets/sft/sft_cleaning.jsonl`，数据统计保存在各目录的 `*_stats.json`。

## 3. SFT 冷启动：2 个 epoch

```bash
uv run python train_sft.py \
    --base-model Qwen/Qwen3.5-4B \
    --epochs 2 \
    --batch-size 4 \
    --rank 32 \
    --learning-rate 2e-5 \
    --seed 42 \
    --run-name spec-o3-sft \
    --swanlab-mode online
```

默认读取 `datasets/sft/`。每个 epoch 完成验证后，在远程保存两份权重，并打印路径：

| 路径 | 后续用途 |
| --- | --- |
| `state_path` | 传给 `train_rl.py --state-path`，继续训练 |
| `sampler_path` | 传给 `eval.py --model-path`，推理评测 |

**保存好 epoch 2 的两种路径。** 也可以从 PyTRIO 权重控制台获取。训练脚本不在本地保存权重文件。

训练 9B 时，使用 `--base-model Qwen/Qwen3.5-9B` 和独立的 `--run-name`，后续 RL 从这份 9B SFT 的 `state_path` 开始。

## 4. 从 SFT state 开始 GRPO：1 个 epoch

把占位符替换成 **PyTRIO 权重控制台获取或训练终端打印的 SFT epoch 2 `state_path`**：

```bash
uv run python train_rl.py \
    --state-path '<SFT epoch 2 的 state_path>' \
    --epochs 1 \
    --batch-size 8 \
    --group-size 8 \
    --learning-rate 1e-6 \
    --max-turns 8 \
    --max-tokens 2048 \
    --max-seq-len 16384 \
    --temperature 1.0 \
    --top-p 1.0 \
    --save-every 100 \
    --seed 42 \
    --run-name spec-o3-rl \
    --swanlab-mode online
```

每批 8 道题，每题采样 8 条轨迹；一个 epoch 共 24,864 条轨迹、389 个 rollout batch。每 100 个累计 rollout step 和每个 epoch 结束时，都会保存并打印 state、sampler 两种路径。

`--max-tokens` 是单轮生成上限；`--max-seq-len` 包含文字、图片和工具历史。`--max-turns 8` 包含最终回答这一轮。PyTRIO 自动拆分请求并累计梯度，无需设置 mini batch。

从 SFT 接 RL 默认使用新的优化器。继续已有 RL 训练时，可换成 RL 的 `state_path` 并加 `--resume-optimizer`；它会恢复优化器状态，但脚本仍从新一轮数据遍历开始，不恢复之前的样本位置。

上面保留首轮 4B 的 1 epoch 命令。要从 SFT state 开始训练 3 个 epoch，改为 `--epochs 3`。RL 会从 `state_path` 恢复基模；4B 与 9B 应分别使用各自的 SFT 权重和 `--run-name`。

## 5. 评测 Base、SFT 和 RL

下面的命令统一使用 6,144 token 的单轮上限和 16,384 token 的总上下文上限。默认评测 `datasets/rl/bench_dev.jsonl` 中的 256 条样本。博客已新增 4B / 9B RL epoch 3 结果，分别答对 175 / 191 题。

### Base

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --output outputs/base-dev-6144 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

### SFT epoch 2

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path '<SFT epoch 2 的 sampler_path>' \
    --output outputs/sft-dev-6144 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

### RL epoch 1

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path '<RL epoch 1 的 sampler_path>' \
    --output outputs/rl-dev-6144 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

### 4B RL epoch 3

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path '<4B RL epoch 3 的 sampler_path>' \
    --output outputs/rl-4b-e3 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

### 9B RL epoch 3

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-9B \
    --model-path '<9B RL epoch 3 的 sampler_path>' \
    --output outputs/rl-9b-e3 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

`--base-model` 必须与 sampler 权重的基模一致；`--model-revision` 可指定 tokenizer 与图像处理器的 HF 文件版本，默认 4B 固定版本、9B 使用 `main`。不传 `--model-path` 就评测 Base，并自动追加最终答案格式提醒；SFT/RL 使用原始提示词。`--output` 必须填写，各模型使用不同目录。

每个输出目录中包含：

- `metrics.json`：评测配置与分类指标。
- `trajectories.jsonl`：逐样本的完整交互记录。
- `images/`：评测过程中工具生成的局部光谱图。

博客已有结果的历史单轮预算为 4B Base 6,132、4B SFT 2,048、三组 RL 6,144 token。这里的 Base/SFT 命令用于统一预算的新评测，结果需单独记录；两个 RL epoch 3 命令与博客新增结果的预算一致。完整测试集可通过 `--data datasets/rl/bench_test.jsonl` 指定；正式使用前需处理统计中记录的 2 个 SFT/test 光谱对象重叠。

## 6. 生成结果图

```bash
uv run python analysis.py
```

输出 `images/results_comparison.png` 和同名矢量 PDF。脚本当前绘制五组固定结果：4B Base、SFT epoch 2、RL epoch 1、RL epoch 3，以及 9B RL epoch 3。四个面板分别展示 Accuracy、Macro F1、严格格式合规率与答对题数；取得新评测结果后，需要同步修改 `analysis.py` 中的数值与图注。

## 7. 查看完整参数

```bash
uv run python prepare_data.py --help
uv run python train_sft.py --help
uv run python train_rl.py --help
uv run python eval.py --help
```
