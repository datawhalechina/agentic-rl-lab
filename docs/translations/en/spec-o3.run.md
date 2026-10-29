# Spec-o3 Quick Start

Follow this order: environment setup → download data → SFT cold start → GRPO → evaluation.

See the [Spec-o3 article](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/spec-o3/readme.md) for the method and results, and [SwanLab](https://swanlab.cn/@kmno4/agentic-rl-lab-spec-o3/v1/rjj3zx/runs) for public training curves.

## 1. Get the project, install dependencies and log in

Install uv first, then run:

```bash
git clone https://github.com/datawhalechina/agentic-rl-lab.git
cd agentic-rl-lab/experiments/spec-o3
uv sync --locked

uv run trio login
uv run swanlab login
```

**Run all subsequent commands inside `experiments/spec-o3/`.** If you already have the repository, enter that directory and start with `uv sync --locked`.

Python 3.13+ is required, with PyTRIO pinned to 0.2.9. Data, tool plotting and orchestration run locally; training and sampling use remote PyTRIO resources. SwanLab logs to the cloud by default. With `--swanlab-mode disabled`, you can skip SwanLab login.

## 2. Download and prepare data

```bash
uv run python prepare_data.py sft
uv run python prepare_data.py bench
```

Complete `sft` before running `bench`: the latter needs the SFT object list to form the development split. Data and images are stored inside the project:

| Path | Purpose |
| --- | --- |
| `datasets/sft/sft_train.jsonl` | SFT training: 848 examples |
| `datasets/sft/sft_val.jsonl` | SFT validation: 94 examples |
| `datasets/sft/images/` | Images from SFT trajectories |
| `datasets/rl/bench_rl_train.jsonl` | RL training: 3,108 examples |
| `datasets/rl/bench_dev.jsonl` | Development evaluation: 256 examples |
| `datasets/rl/bench_test.jsonl` | Full test set: 6,754 examples |
| `datasets/rl/images/` | Initial spectrum images |
| `datasets/rl/spectra/` | Spectral arrays used by plotting tools |

Rerunning reuses downloaded files. Prepared image and spectrum paths are absolute, so prepare the data again after moving the directory or changing machines. Text-cleaning records are in `datasets/sft/sft_cleaning.jsonl`; dataset statistics are in each directory's `*_stats.json`.

## 3. SFT cold start: 2 epochs

```bash
uv run python train_sft.py \
    --base-model Qwen/Qwen3.5-4B \
    --epochs 2 \
    --batch-size 4 \
    --rank 32 \
    --learning-rate 2e-5 \
    --seed 42 \
    --run-name spec-o3-sft \
    --swanlab-mode online
```

The default input is `datasets/sft/`. After validation at each epoch, two kinds of remote weights are saved and their paths printed:

| Path | Later use |
| --- | --- |
| `state_path` | Pass to `train_rl.py --state-path` to continue training |
| `sampler_path` | Pass to `eval.py --model-path` for inference and evaluation |

**Keep both paths from epoch 2.** They are also available in the PyTRIO weights console. The script does not save model weights locally.

For 9B, use `--base-model Qwen/Qwen3.5-9B` and a separate `--run-name`. Start its subsequent RL run from the corresponding 9B SFT `state_path`.

## 4. Start GRPO from the SFT state: 1 epoch

Replace the placeholder with the **SFT epoch-2 `state_path` from the PyTRIO weights console or training terminal**:

```bash
uv run python train_rl.py \
    --state-path '<SFT epoch 2 的 state_path>' \
    --epochs 1 \
    --batch-size 8 \
    --group-size 8 \
    --learning-rate 1e-6 \
    --max-turns 8 \
    --max-tokens 2048 \
    --max-seq-len 16384 \
    --temperature 1.0 \
    --top-p 1.0 \
    --save-every 100 \
    --seed 42 \
    --run-name spec-o3-rl \
    --swanlab-mode online
```

Each batch contains 8 questions with 8 trajectories per question. One epoch therefore contains 24,864 trajectories and 389 rollout batches. State and sampler paths are saved and printed every 100 cumulative rollout steps and at the end of each epoch.

`--max-tokens` limits a single generation turn; `--max-seq-len` includes text, images and tool history. `--max-turns 8` includes the final-answer turn. PyTRIO splits requests and accumulates gradients automatically; no mini-batch setting is needed.

The transition from SFT to RL uses a fresh optimizer by default. To continue an RL run, use its `state_path` and add `--resume-optimizer`. This restores optimizer state, but starts a new data pass rather than restoring the previous sample position.

The command above preserves the first 4B run's 1-epoch configuration. For 3 epochs from the SFT state, use `--epochs 3`. RL restores the base model from `state_path`; use each model's own SFT weights and `--run-name` for 4B and 9B.

## 5. Evaluate Base, SFT and RL

The following commands use a uniform per-turn budget of 6,144 tokens and total context limit of 16,384 tokens. By default they evaluate the 256 examples in `datasets/rl/bench_dev.jsonl`. The article includes 4B and 9B RL epoch-3 results with 175 and 191 correct answers respectively.

### Base

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --output outputs/base-dev-6144 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

### SFT epoch 2

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path '<SFT epoch 2 的 sampler_path>' \
    --output outputs/sft-dev-6144 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

### RL epoch 1

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path '<RL epoch 1 的 sampler_path>' \
    --output outputs/rl-dev-6144 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

### 4B RL epoch 3

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path '<4B RL epoch 3 的 sampler_path>' \
    --output outputs/rl-4b-e3 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

### 9B RL epoch 3

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-9B \
    --model-path '<9B RL epoch 3 的 sampler_path>' \
    --output outputs/rl-9b-e3 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

`--base-model` must match the sampler weights' base model. `--model-revision` selects the Hugging Face revision for the tokenizer and image processor; the default pins a revision for 4B and uses `main` for 9B. Omitting `--model-path` evaluates Base and appends an answer-format reminder; SFT/RL use the original prompt. `--output` is required, with a separate directory for each model.

Each output directory contains:

- `metrics.json`: evaluation configuration and classification metrics.
- `trajectories.jsonl`: complete per-example interaction records.
- `images/`: local spectrum plots produced by tools during evaluation.

The article's historical per-turn budgets were 6,132 tokens for 4B Base, 2,048 for 4B SFT and 6,144 for the three RL runs. The Base/SFT commands here create new evaluations at a uniform budget; record their results separately. Both RL epoch-3 commands match the budgets of the newly reported results. Select the full test set with `--data datasets/rl/bench_test.jsonl`; before formal use, resolve the 2 overlapping SFT/test spectral objects recorded in the statistics.

## 6. Generate result figures

```bash
uv run python analysis.py
```

This writes `images/results_comparison.png` and a vector PDF with the same basename. The script currently plots five fixed results: 4B Base, SFT epoch 2, RL epoch 1, RL epoch 3, and 9B RL epoch 3. Its panels show accuracy, Macro F1, strict format compliance and correct-answer counts. Update the values and captions in `analysis.py` after new evaluations.

## 7. Inspect all parameters

```bash
uv run python prepare_data.py --help
uv run python train_sft.py --help
uv run python train_rl.py --help
uv run python eval.py --help
```
