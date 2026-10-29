# DAPO 快速启动

以下命令均从仓库根目录运行。项目要求 Python `>=3.13`。

```bash
uv sync
trio login
swanlab login
```

## 1. 下载数据

```bash
uv run python experiments/dapo/prepare_data.py
```

这条命令会准备 DAPO-Math 训练集和 AIME25 评测集。

## 2. 启动训练

```bash
uv run python experiments/dapo/train.py \
    --algorithm dapo \
    --max-steps 10 \
    --groups-per-step 4 \
    --group-size 8 \
    --max-candidate-multiplier 2 \
    --max-prompt-tokens 512 \
    --max-tokens 4096 \
    --overlong-cache 1024 \
    --swanlab-mode online
```

同一入口也支持 `--algorithm grpo`，可用于 matched baseline。

## 3. 运行评测

先评测 Base Model：

```bash
uv run python experiments/dapo/eval.py
```

再评测训练得到的 sampler weights：

```bash
uv run python experiments/dapo/eval.py \
    --model-path 'trio://RUN_ID/sampler_weights/DAPO_WEIGHTS_NAME' \
    --output experiments/dapo/eval-results/aime25-dapo-step10.jsonl
```
