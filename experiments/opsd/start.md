# OPSD 快速启动

以下命令均从仓库根目录运行。项目要求 Python `>=3.13`。

```bash
uv sync
trio login
swanlab login
```

## 1. 下载数据

```bash
uv run python experiments/opsd/00-datasets.py
```

这条命令会准备 OPSD 训练集和 AIME25 评测集。

## 2. 启动训练

```bash
uv run python experiments/opsd/01-opsd-async.py \
    --steps 100 \
    --batch-size 32 \
    --group-size 1 \
    --max-tokens 1024 \
    --sample-size 0 \
    --save-every-steps 25 \
    --max-concurrency 32 \
    --swanlab-mode online
```

训练日志会打印可续训的 state 路径和用于采样、评测的 sampler weights 路径。

## 3. 运行评测

先评测 Base Model：

```bash
uv run python experiments/opsd/00-eval-aime25.py \
    --val-n 12 \
    --max-tokens 38912 \
    --temperature 1.0 \
    --enable-thinking false \
    --output experiments/opsd/eval-results/aime25-base.jsonl
```

再评测训练得到的 sampler weights：

```bash
uv run python experiments/opsd/00-eval-aime25.py \
    --val-n 12 \
    --max-tokens 38912 \
    --temperature 1.0 \
    --enable-thinking false \
    --model-path 'trio://RUN_ID/sampler_weights/OPSD_WEIGHTS_NAME' \
    --output experiments/opsd/eval-results/aime25-opsd-step100.jsonl
```
