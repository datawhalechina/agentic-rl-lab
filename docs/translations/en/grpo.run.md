# GRPO Quick Start

Run all commands from the repository root. The project requires Python `>=3.13`.

```bash
uv sync
trio login
swanlab login
```

## 1. Prepare the data

No separate download command is needed. The first training run downloads the `openai/gsm8k` train split and uses the local Hugging Face cache.

## 2. Start training

Synchronous version example:

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode online
```

`02-demo-async.py` has the same logic as the synchronous version, except that it submits the rollouts for each question in a batch concurrently. It already adapts to the current PyTRIO 0.2.9 configuration parameters and can be run directly.

## 3. Evaluate the run

This chapter currently has no separate `eval.py`. Training records `reward`, `frac_degenerate`, `rollout/avg_gen_len`, `train_tokens` and `loss_mean` in the terminal and SwanLab, then prints the saved sampler weights path.

When comparing losses, keep the other settings fixed and change only:

```bash
--loss-fn importance_sampling
--loss-fn ppo
--loss-fn cispo
```
