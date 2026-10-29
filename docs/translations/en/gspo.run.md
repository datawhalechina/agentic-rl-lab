# GSPO Quick Start

Run all commands from the repository root. The project requires Python `>=3.13`.

```bash
uv sync
trio login
swanlab login
```

## 1. Download the data

```bash
uv run python experiments/gspo/prepare_data.py
```

This prepares the DAPO-Math training set and AIME25 evaluation set.

## 2. Start training

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

The training log prints a state path for resuming training and a sampler weights path for sampling and evaluation.

## 3. Run evaluation

First evaluate the base model:

```bash
uv run python experiments/gspo/eval.py
```

Then evaluate the trained sampler weights:

```bash
uv run python experiments/gspo/eval.py \
    --model-path 'trio://RUN_ID/sampler_weights/GSPO_WEIGHTS_NAME' \
    --output experiments/gspo/eval-results/aime25-gspo-step100.jsonl
```
