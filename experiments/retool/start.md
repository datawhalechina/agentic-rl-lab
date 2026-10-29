# ReTool 快速启动

以下命令均从仓库根目录运行。项目要求 Python `>=3.13`。

```bash
uv sync
trio login
swanlab login
```

## 1. 下载数据

准备训练集与 AIME25 评测集：

```bash
uv run python experiments/retool/prepare_data.py
uv run python experiments/opsd/00-datasets.py --only aime25
```

## 2. 启动训练

先运行 20-step 验证实验：

```bash
uv run python experiments/retool/train.py \
    --max-steps 20 \
    --save-every 5 \
    --run-name retool-qwen35-4b-step20
```

完整实验把 `--max-steps` 改为 `200`，把 `--save-every` 改为 `50`。

## 3. 运行评测

先评测 Base Model：

```bash
uv run python experiments/retool/eval.py \
    --mode retool \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --output experiments/retool/eval-results/aime25-retool-base.jsonl
```

再评测训练得到的 sampler weights：

```bash
uv run python experiments/retool/eval.py \
    --mode retool \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --model-path 'trio://RUN_ID/sampler_weights/RETOOL_WEIGHTS_NAME' \
    --output experiments/retool/eval-results/aime25-retool-step200.jsonl
```
