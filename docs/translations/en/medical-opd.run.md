# Medical OPD Quick Start

Run all commands from the repository root. The project requires Python `>=3.13`.

```bash
uv sync
trio login
swanlab login
```

Evaluation uses TRIO's OpenAI-compatible API. If the configuration file does not yet exist, run:

```bash
cp experiments/medical-opd/.env.example experiments/medical-opd/.env
```

Then fill in `PYTRIO_API_KEY` in `experiments/medical-opd/.env`.

## 1. Download the data

```bash
uv run python experiments/medical-opd/00-download-dataset.py
```

This prepares the Medical SFT, MedQA-zh and C-Eval data.

## 2. Start training

First train the Medical SFT teacher:

```bash
uv run python experiments/medical-opd/02-medical-sft.py \
    --num-epochs 3 \
    --batch-size 16 \
    --max-length 2048 \
    --swanlab-mode online
```

Record the SFT `sampler_weights` path printed at the end, then start Medical OPD:

```bash
uv run python experiments/medical-opd/03-medical-opd-async.py \
    --teacher-model-path 'trio://RUN_ID/sampler_weights/SFT_WEIGHTS_NAME' \
    --steps 300 \
    --batch-size 4 \
    --group-size 4 \
    --sample-size 0 \
    --max-tokens 2048 \
    --learning-rate 4e-5 \
    --save-every-steps 300 \
    --swanlab-mode online
```

Replace `RUN_ID` and `SFT_WEIGHTS_NAME` with the actual values from the previous run.

## 3. Run evaluation

Replace the model path below with the sampler weights path printed by OPD training:

```bash
uv run python experiments/medical-opd/01-eval-medical.py \
    --model 'trio://RUN_ID/sampler_weights/OPD_WEIGHTS_NAME' \
    --max-tokens 1024 \
    --concurrency 16

uv run python experiments/medical-opd/01-eval-ceval.py \
    --model 'trio://RUN_ID/sampler_weights/OPD_WEIGHTS_NAME' \
    --max-tokens 8192 \
    --concurrency 16
```

Per-question results and aggregate metrics are written to `experiments/medical-opd/eval-results/`.
