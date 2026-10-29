# DAPO Quick Start

Run all commands from the repository root. The project requires Python `>=3.13`.

```bash
uv sync
trio login
swanlab login
```

## 1. Download the data

```bash
uv run python experiments/dapo/prepare_data.py
```

This prepares the DAPO-Math training set and AIME25 evaluation set.

## 2. Start training

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

The same entry point supports `--algorithm grpo` for a matched baseline.

## 3. Run evaluation

First evaluate the base model:

```bash
uv run python experiments/dapo/eval.py
```

Then evaluate the trained sampler weights:

```bash
uv run python experiments/dapo/eval.py \
    --model-path 'trio://RUN_ID/sampler_weights/DAPO_WEIGHTS_NAME' \
    --output experiments/dapo/eval-results/aime25-dapo-step10.jsonl
```
