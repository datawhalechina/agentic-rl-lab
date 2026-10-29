# GeoQA Vision GRPO Quick Start

> See the [Vision GRPO article](./readme.md) for the method, results and implementation boundaries.

Run all commands from the repository root. Python 3.13 or later is required. Training and evaluation use the remote PyTRIO service.

> **Experiment cost**
>
> `train.py` and `eval.py` consume remote sampling or training resources. Start with the 1-step configuration to verify images, token alignment and account access before launching a full experiment.

## 1. Install dependencies and log in

```bash
uv sync
trio login
swanlab login
```

Training uses online SwanLab tracking by default. For debugging, pass `--swanlab-mode disabled`.

## 2. Download and split GeoQA

```bash
uv run python experiments/vision-grpo/download-dataset.py
```

The default output is:

```text
experiments/vision-grpo/datasets/train.parquet  # 3,503 条
experiments/vision-grpo/datasets/test.parquet   # 固定 100 条
```

To pin a dataset revision:

```bash
uv run python experiments/vision-grpo/download-dataset.py \
  --revision DATASET_COMMIT
```

## 3. Check commands and local imports

```bash
uv run python experiments/vision-grpo/download-dataset.py --help
uv run python experiments/vision-grpo/train.py --help
uv run python experiments/vision-grpo/eval.py --help
uv run python experiments/vision-grpo/analysis.py --help
```

## 4. Run a small 1-step check

```bash
uv run python experiments/vision-grpo/train.py \
  --steps 1 \
  --batch-size 1 \
  --group-size 4 \
  --max-tokens 256 \
  --save-every 0 \
  --no-save-weights \
  --show-samples \
  --swanlab-mode disabled
```

This checks sampling with real images, `input_tokens` alignment, boxed-answer rewards and group advantages. All four answers to a question may receive the same score. Such a group is skipped, leaving `train_datums` at 0 in the log.

## 5. Run a small 20-step experiment

```bash
uv run python experiments/vision-grpo/train.py \
  --steps 20 \
  --batch-size 8 \
  --group-size 8 \
  --max-tokens 1024 \
  --learning-rate 4e-5 \
  --save-every 10 \
  --swanlab-mode online
```

## 6. Launch the full 100-step run

```bash
uv run python experiments/vision-grpo/train.py \
  --base-model Qwen/Qwen3.5-4B \
  --lora-rank 32 \
  --steps 100 \
  --batch-size 8 \
  --group-size 8 \
  --max-tokens 1024 \
  --temperature 1.0 \
  --top-p 1.0 \
  --learning-rate 4e-5 \
  --save-every 25 \
  --experiment-name vision-grpo-qwen35-4b-geoqa \
  --weights-name vision-grpo-qwen35-4b-geoqa \
  --swanlab-mode online
```

The following are saved every 25 steps:

```text
trio://.../sampler_weights/...-sampler
trio://.../training_state/...-state
```

Use `sampler_weights` for evaluation and `training_state` to resume training.

## 7. Evaluate the base model

```bash
uv run python experiments/vision-grpo/eval.py \
  --base-model Qwen/Qwen3.5-4B \
  --max-tokens 1024 \
  --limit 100 \
  --output experiments/vision-grpo/eval-results-base.json
```

## 8. Evaluate a checkpoint

Replace `--model-path` with the sampler weights path printed in the training log:

```bash
uv run python experiments/vision-grpo/eval.py \
  --base-model Qwen/Qwen3.5-4B \
  --model-path 'trio://RUN_ID/sampler_weights/WEIGHTS_NAME' \
  --max-tokens 1024 \
  --limit 100 \
  --output experiments/vision-grpo/eval-results-step-100.json
```

Evaluate one model at a time, using separate output files for the base model and each checkpoint.

## 9. Generate the documented result figures

```bash
uv run python experiments/vision-grpo/analysis.py
```

Output location:

```text
experiments/vision-grpo/images/eval-comparison.png
```

The current `analysis.py` plots the documented base results `71.0% / 75.0%` and step-100 results `87.0% / 91.0%`. After a new experiment, update `BASE_RESULTS` and `GRPO_RESULTS` first.
