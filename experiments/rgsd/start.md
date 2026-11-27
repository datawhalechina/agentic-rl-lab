# RGSD 快速启动

复现 [Rubric-Guided Self-Distillation](https://arxiv.org/abs/2606.12507)：同一个基础模型，Student 只看问题，冻结的 Teacher 额外看到 rubric，Student 在自己的 rollout 上逐 token 向 Teacher 蒸馏。训练全程不调用 judge。

以下命令均从仓库根目录运行。先安装依赖、登录并准备医学数据：

```bash
uv sync
uv run trio login
uv run swanlab login
uv run python experiments/rgsd/prepare_data.py --domains medical
```

数据来自 `sojuL/RubricHub_v1`（固定 revision），每个领域按 seed 42 切出 300 题评测集，其余作训练集。公开版没有官方 train/val 划分，行数也与论文不同，所以这份划分不是论文的那一份。

## 1. 冒烟测试

用 batch 8 跑 2 步，确认梯度范数、一致性和探针检查都正常。`--warmup-steps 2` 让前两步学习率与正式训练一致，分别为 2e-5、4e-5：

```bash
uv run python experiments/rgsd/train.py \
    --domain medical \
    --steps 2 \
    --batch-size 8 \
    --warmup-steps 2 \
    --run-name rgsd-smoke \
    --swanlab-mode disabled
```

## 2. 完整训练

使用与 readme 中一致的配置：20 步、batch 128，约 2 小时。

```bash
uv run python experiments/rgsd/train.py \
    --domain medical \
    --steps 20 \
    --batch-size 128 \
    --max-tokens 2048 \
    --learning-rate 4e-5 \
    --save-every-steps 10 \
    --max-concurrency 64 \
    --max-truncated-rate 0.4 \
    --run-name rgsd-stageA-medical-lr4e-5-v3 \
    --swanlab-mode online
```

结果保存在 `experiments/rgsd/train-results/rgsd-stageA-medical-lr4e-5-v3/`：`rollouts.jsonl` 记录每步回答，`checkpoints.jsonl` 记录第 10、20 步的 state 和采样权重路径。同名目录已有训练结果时会拒绝新建训练。

## 3. 评测

评测分两步：先生成，再用 judge 打分。换 judge 时不需要重新生成。

```bash
uv run python experiments/rgsd/eval.py generate --domain medical --tag base
uv run python experiments/rgsd/eval.py generate --domain medical \
    --run-dir experiments/rgsd/train-results/rgsd-stageA-medical-lr4e-5-v3
```

把 `experiments/rgsd/.env.example` 复制为 `.env` 并填写 judge，随后对上述生成结果打分：

```bash
uv run python experiments/rgsd/eval.py judge --judge-backend chat \
    experiments/rgsd/eval-results/generations-medical-eval-nothink-base.jsonl \
    experiments/rgsd/eval-results/generations-medical-eval-nothink-rgsd-stageA-medical-lr4e-5-v3-step*.jsonl
uv run python experiments/rgsd/analysis.py
```

打印逐步训练指标、Base 与 checkpoint 的得分和配对差值（含等长校正），并生成 `images/training-dynamics.png` 和 `images/eval-comparison.png`。

分析脚本与 chat 评测共用 `experiments/rgsd/.env` 中的 `RGSD_JUDGE_MODEL_NAME`，自动读取对应裁判的评分文件，无需额外传参。更换裁判后，先重新打分，再运行上述分析命令。分析过程只读取模型名称，不调用裁判接口，也不需要 API Key。
