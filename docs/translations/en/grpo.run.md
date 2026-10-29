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

With the current dependencies, use the synchronous entry point:

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode online
```

`02-demo-async.py` still contains a timeout setting from an older PyTRIO version. It fails at startup until that setting is updated, so this quick start uses the synchronous script.

## 3. Evaluate the run

This chapter currently has no separate `eval.py`. Training records `reward`, `frac_degenerate`, `rollout/avg_gen_len`, `train_tokens` and `loss_mean` in the terminal and SwanLab, then prints the saved sampler weights path.

When comparing losses, keep the other settings fixed and change only:

```bash
--loss-fn importance_sampling
--loss-fn ppo
--loss-fn cispo
```
