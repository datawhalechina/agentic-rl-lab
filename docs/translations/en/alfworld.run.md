# ALFWorld Agentic RL Quick Start

> See the [ALFWorld Agentic RL article](./readme.md) for the method, training results and code walkthrough.

Run all commands from the repository root. The project requires Python `>=3.13`.

> **Experiment cost**
>
> The full configuration samples 64 long trajectories per update, with up to 12K tokens per sequence. The 80-step experiment documented here cost ¥1613.71 on PyTRIO. Run the 1-step check before launching full training.

## 1. Install dependencies and log in

ALFWorld uses an optional dependency group, separate from the other experiments:

```bash
uv sync --extra alfworld
trio login
swanlab login
```

## 2. Download ALFWorld data

```bash
uv run --extra alfworld alfworld-download \
    --data-dir "$PWD/experiments/alfworld/datasets/alfworld"
```

Training and evaluation read this directory by default; no additional `--data-root` is needed.

## 3. Run a 1-step check

Use one game and 8 trajectories to check data, environments, sampling, rewards, PPO and checkpointing:

```bash
uv run --extra alfworld python experiments/alfworld/train.py \
    --max-steps 1 \
    --games-per-batch 1 \
    --group-size 8 \
    --max-episode-steps 10 \
    --save-every 0 \
    --run-name alfworld-smoke \
    --swanlab-mode disabled
```

`--save-every 0` disables intermediate checkpoints only. The final state and sampler weights are still saved when training finishes.

## 4. Launch the full 80-step run

```bash
uv run --extra alfworld python experiments/alfworld/train.py \
    --base-model Qwen/Qwen3.5-4B \
    --max-steps 80 \
    --games-per-batch 8 \
    --group-size 8 \
    --max-episode-steps 50 \
    --max-trajectory-tokens 12000 \
    --max-assistant-tokens 2048 \
    --temperature 1.0 \
    --top-p 1.0 \
    --learning-rate 1e-6 \
    --save-every 20 \
    --run-name alfworld-agent-rl-qwen35-4b \
    --swanlab-mode online
```

Two kinds of paths are printed every 20 updates:

```text
Saved state: trio://.../training_state/...
Saved sampler weights: trio://.../sampler_weights/...
```

Use `Saved sampler weights` for checkpoint evaluation, not the state path.

## 5. Evaluate the base model

Full evaluation covers `valid_seen=140` and `valid_unseen=134`, for 274 games in total:

```bash
uv run --extra alfworld python experiments/alfworld/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/alfworld/eval_results/base-qwen35-4b-eval.jsonl \
    --swanlab-mode disabled
```

## 6. Evaluate a checkpoint

Replace `--model-path` with the sampler weights path from the training log. This example uses Step 80:

```bash
uv run --extra alfworld python experiments/alfworld/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path 'trio://RUN_ID/sampler_weights/WEIGHTS_NAME' \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/alfworld/eval_results/checkpoint-80steps.jsonl \
    --swanlab-mode disabled
```

By default, `eval.py` refuses to overwrite an existing JSONL. To rerun evaluation at the same path, explicitly add:

```bash
--overwrite-output
```

## 7. Generate evaluation figures

By default, `analysis.py` reads these three files:

```text
experiments/alfworld/eval_results/base-qwen35-4b-eval.jsonl
experiments/alfworld/eval_results/checkpoint-40steps.jsonl
experiments/alfworld/eval_results/checkpoint-80steps.jsonl
```

Once all three files are ready, run:

```bash
uv run python experiments/alfworld/analysis.py
```

Figures are saved to:

```text
experiments/alfworld/images/alfworld_checkpoint_evaluation.png
```
