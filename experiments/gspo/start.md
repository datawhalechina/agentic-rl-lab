# GSPO 快速启动

以下命令均从仓库根目录运行。项目要求 Python `>=3.13`。

```bash
uv sync
trio login
swanlab login
```

## 1. 下载数据

```bash
uv run python experiments/gspo/prepare_data.py
```

这条命令会准备 DAPO-Math 训练集和 AIME25 评测集。

## 2. 启动训练

```bash
uv run python experiments/gspo/train.py \
    --max-steps 100 \
    --groups-per-step 8 \
    --group-size 8 \
    --max-prompt-tokens 2048 \
    --max-tokens 4096 \
    --save-every 20 \
    --swanlab-mode online
```

训练日志会打印可续训的 state 路径和用于采样、评测的 sampler weights 路径。

## 3. 运行评测

先评测 Base Model：

```bash
uv run python experiments/gspo/eval.py
```

再评测训练得到的 sampler weights：

```bash
uv run python experiments/gspo/eval.py \
    --model-path 'trio://RUN_ID/sampler_weights/GSPO_WEIGHTS_NAME' \
    --output experiments/gspo/eval-results/aime25-gspo-step100.jsonl
```
