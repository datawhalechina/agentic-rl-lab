# AgentOPSD Quick Start

> See the [AgentOPSD article](./readme.md) for the method, results and code walkthrough.

Run all commands from the repository root. Python `>=3.13` is required. ALFWorld environments, trajectory orchestration and turn credit run locally; PyTRIO handles sampling and LoRA training remotely.

> **Experiment cost**
>
> The full configuration samples `16 × 8 = 128` long trajectories per update, or 10,240 rollouts over 80 steps, and also has the teacher rescore those trajectories. The documented run cost ¥675.69 on PyTRIO. Complete the 1-step smoke check before launching full training.

## 1. Install dependencies and log in

```bash
uv sync --extra alfworld
trio login
swanlab login
```

`swanlab login` is needed only with `--swanlab-mode online`. Check the key dependency versions:

```bash
uv run --extra alfworld python -c 'from importlib.metadata import version; print("pytrio", version("pytrio")); print("alfworld", version("alfworld")); print("swanlab", version("swanlab"))'
```

This experiment pins PyTRIO `0.2.8`, ALFWorld `0.4.2` and SwanLab `0.9.2`.

## 2. Download ALFWorld data

```bash
uv run --extra alfworld alfworld-download \
    --data-dir "$PWD/experiments/agentopsd/datasets/alfworld"
```

Training and evaluation use this directory by default; no additional `--data-root` is needed.

## 3. Inspect available parameters

```bash
uv run --extra alfworld python experiments/agentopsd/train.py --help
uv run --extra alfworld python experiments/agentopsd/eval.py --help
uv run python experiments/agentopsd/analysis.py --help
```

## 4. Run a 1-step smoke check

Use one game and 4 trajectories to check the ALFWorld environment, student rollouts, teacher rescoring from the same snapshot, turn credit, PPO and checkpointing:

```bash
uv run --extra alfworld python experiments/agentopsd/train.py \
    --max-steps 1 \
    --tasks-per-update 1 \
    --group-size 4 \
    --max-turns 20 \
    --teacher-concurrency 4 \
    --save-every 1 \
    --seed 3 \
    --run-name agentopsd-smoke \
    --swanlab-mode disabled
```

All 4 trajectories may succeed or fail, making group-relative advantages zero. The script retains rollout metrics and skips the parameter update. Increase `--tasks-per-update` when a reliable backward-pass check is needed.

## 5. Launch the full 80-step run

```bash
uv run --extra alfworld python experiments/agentopsd/train.py \
    --base-model Qwen/Qwen3.5-4B \
    --max-steps 80 \
    --tasks-per-update 16 \
    --group-size 8 \
    --max-turns 50 \
    --max-trajectory-tokens 14336 \
    --max-action-tokens 512 \
    --temperature 1.0 \
    --top-p 1.0 \
    --teacher-concurrency 16 \
    --gamma 0.95 \
    --reshape-lambda 0.5 \
    --weight-bound 0.2 \
    --ppo-clip-low 0.8 \
    --ppo-clip-high 1.24 \
    --lora-rank 32 \
    --learning-rate 4e-6 \
    --save-every 40 \
    --seed 42 \
    --run-name agentopsd-alfworld-qwen35-4b \
    --swanlab-project agentic-rl-lab-agentopsd \
    --swanlab-mode online
```

Two kinds of paths are printed at Step 40, Step 80 and the end of training:

```text
Saved state: trio://.../training_state/...
Saved sampler weights: trio://.../sampler_weights/...
```

`eval.py` uses the `Saved sampler weights` path.

## 6. Check the evaluation pipeline

This evaluates only 2 seen and 2 unseen games to check output paths and environments:

```bash
uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --split all \
    --limit-per-split 2 \
    --games-per-batch 2 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/base-smoke.jsonl \
    --swanlab-mode disabled
```

## 7. Evaluate the complete base model split

Full evaluation covers `valid_seen=140` and `valid_unseen=134`, for 274 games in total:

```bash
uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/base-qwen35-4b-eval.jsonl \
    --swanlab-mode disabled
```

## 8. Evaluate your checkpoint

Replace `--model-path` with the sampler weights path printed in the training log:

```bash
uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path 'trio://RUN_ID/sampler_weights/WEIGHTS_NAME' \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/checkpoint-eval.jsonl \
    --swanlab-mode disabled
```

By default, `eval.py` protects existing JSONL files. To rerun evaluation into the same output file, explicitly add:

```bash
--overwrite-output
```

## 9. Reproduce the Step 40 and Step 80 evaluations

```bash
uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path 'trio://run_sschmbprfwg0/sampler_weights/agentopsd-alfworld-qwen35-4b-step-40-weights' \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/checkpoint-eval-step40.jsonl \
    --swanlab-mode disabled

uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path 'trio://run_sschmbprfwg0/sampler_weights/agentopsd-alfworld-qwen35-4b-step-80-weights' \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/checkpoint-eval-step80.jsonl \
    --swanlab-mode disabled
```

## 10. Generate checkpoint comparison figures

By default, `analysis.py` reads:

```text
experiments/agentopsd/eval_results/base-qwen35-4b-eval.jsonl
experiments/agentopsd/eval_results/checkpoint-eval-step40.jsonl
experiments/agentopsd/eval_results/checkpoint-eval-step80.jsonl
```

Once the three files are ready, run:

```bash
uv run python experiments/agentopsd/analysis.py
```

Figures are saved to:

```text
experiments/agentopsd/images/agentopsd_checkpoint_evaluation.png
```

To customize the result directory, output location or resolution:

```bash
uv run python experiments/agentopsd/analysis.py \
    --result-dir experiments/agentopsd/eval_results \
    --output experiments/agentopsd/images/agentopsd_checkpoint_evaluation.png \
    --dpi 300
```
