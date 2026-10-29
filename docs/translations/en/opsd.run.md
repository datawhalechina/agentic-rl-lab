# OPSD Quick Start

Run all commands from the repository root. The project requires Python `>=3.13`.

```bash
uv sync
trio login
swanlab login
```

## 1. Download the data

```bash
uv run python experiments/opsd/00-datasets.py
```

This prepares the OPSD training set and AIME25 evaluation set.

## 2. Start training

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

The training log prints a state path for resuming training and a sampler weights path for sampling and evaluation.

## 3. Run evaluation

First evaluate the base model:

```bash
uv run python experiments/opsd/00-eval-aime25.py \
    --val-n 12 \
    --max-tokens 38912 \
    --temperature 1.0 \
    --enable-thinking false \
    --output experiments/opsd/eval-results/aime25-base.jsonl
```

Then evaluate the trained sampler weights:

```bash
uv run python experiments/opsd/00-eval-aime25.py \
    --val-n 12 \
    --max-tokens 38912 \
    --temperature 1.0 \
    --enable-thinking false \
    --model-path 'trio://RUN_ID/sampler_weights/OPSD_WEIGHTS_NAME' \
    --output experiments/opsd/eval-results/aime25-opsd-step100.jsonl
```
