# Tournament-GRPO Quick Start

Run all commands below from the repository root with Python `>=3.13`. See [readme.md](./readme.md) for the algorithm explanation.

## 0. Install and log in

```bash
uv sync
uv run trio login
uv run swanlab login
```

## 1. Prepare the data

```bash
uv run python experiments/tournament-grpo/prepare_data.py
```

The script downloads the training data and Deep Research Bench from a pinned commit of the authors' repository, then uses the `Qwen/Qwen3.5-4B` tokenizer to filter out questions exceeding 4096 tokens:

```text
experiments/tournament-grpo/datasets/
├── raw/                      # 作者原始 Parquet
├── train.jsonl               # 8,378 条
├── val.jsonl                 # 440 条
└── deep_research_bench/      # query / criteria / reference 各 100 条
```

## 2. Configure API keys

```bash
cp -n experiments/tournament-grpo/.env.example experiments/tournament-grpo/.env
```

Configure search, the training judge, and the evaluation judge in `experiments/tournament-grpo/.env`:

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
JINA_API_KEY=
JINA_READER_BASE_URL=https://r.jina.ai

TOURNAMENT_JUDGE_BACKEND=jev
TOURNAMENT_JUDGE_API_URL=https://api.typesafe.ai/v1/systemone
TOURNAMENT_JUDGE_MODEL_NAME=jev-1.14.0
TOURNAMENT_JUDGE_API_KEY=your_jev_api_key

RACE_JUDGE_API_URL=https://api.deepseek.com/v1/responses
RACE_JUDGE_MODEL_NAME=deepseek-flash
RACE_JUDGE_API_KEY=your_deepseek_api_key
```

- `JINA_API_KEY` can be left empty, in which case the rate limit is 20 RPM. In mainland China, you can change `JINA_READER_BASE_URL` to `https://r.jinaai.cn`.
- The official Jev endpoint cannot be accessed directly from mainland China. Tool and judge requests do not read the shell's `HTTP(S)_PROXY`; readers in mainland China need to enable a system-wide global proxy or change `TOURNAMENT_JUDGE_API_URL` to a compatible service accessible from mainland China.
- In `.env`, you can also set `DEEPSEEK_SEARCH_PROXY` and `JINA_READER_PROXY` separately for search and Jina.

Check whether Jev is available:

```bash
uv run python experiments/tournament-grpo/judge.py
```

## 3. Run a single-step check

```bash
uv run python experiments/tournament-grpo/train.py \
  --max-steps 1 \
  --questions-per-batch 1 \
  --group-size 8 \
  --run-name tournament-grpo-smoke
```

This step samples 8 trajectories for 1 question, runs 28 judge comparisons, and updates the weights once, taking approximately 2–5 minutes. After the run, check `experiments/tournament-grpo/outputs/tournament-grpo-smoke/train.jsonl`: `judge/comparisons` should be 28 and `reward/has_answer` should be greater than 0.

## 4. Run 20-step training

```bash
uv run python experiments/tournament-grpo/train.py \
  --max-steps 20 \
  --questions-per-batch 8 \
  --group-size 8 \
  --learning-rate 4e-5 \
  --save-every 10 \
  --run-name tournament-grpo-20step
```

Each step contains 64 trajectories and 224 judge comparisons. Training stops if judge requests still fail after 3 retries. Once the endpoint is fixed, resume from the latest checkpoint:

```bash
uv run python experiments/tournament-grpo/train.py \
  --max-steps 20 \
  --resume experiments/tournament-grpo/outputs/tournament-grpo-20step/checkpoint-step-10.json \
  --run-name tournament-grpo-20step-resumed
```

Training outputs are saved in `experiments/tournament-grpo/outputs/<run-name>/`:

```text
train.jsonl                 # 每步指标
judge.jsonl                 # 每场比赛的候选、胜者和裁判输出
rollout-step-*.jsonl        # 第 1 步和 checkpoint 步的完整轨迹
checkpoint-step-*.json      # 续训 state 和采样权重
```

## 5. Evaluate on Deep Research Bench

First generate research reports for Base and Step 20:

```bash
uv run python experiments/tournament-grpo/eval.py generate \
  --batch-size 16
uv run python experiments/tournament-grpo/eval.py generate \
  --batch-size 16 \
  --checkpoint experiments/tournament-grpo/outputs/tournament-grpo-20step/checkpoint-step-20.json
```

Then score them with RACE:

```bash
uv run python experiments/tournament-grpo/eval.py score \
  --judge-concurrency 8
uv run python experiments/tournament-grpo/eval.py score \
  --judge-concurrency 8 \
  --checkpoint experiments/tournament-grpo/outputs/tournament-grpo-20step/checkpoint-step-20.json
```

Evaluation uses 100 questions, with 1 trajectory generated greedily per question. Both generation and scoring save results question by question; rerun the same command to resume after an interruption. You can first add `--limit 4` to check the workflow on just the first 4 questions.

Results are saved separately by policy model, weights, and judge model:

```text
experiments/tournament-grpo/outputs/drb/Qwen--Qwen3.5-4B/
├── base/
│   ├── articles.jsonl
│   ├── generation-summary.json
│   └── race/deepseek-flash/race-summary.json
└── trio--run_<ID>--sampler_weights--step-20-weights/
    └── ...
```

## 6. View the results

View training curves in the `tournament-grpo` project on SwanLab. Focus on `reward/*` and `rollout/*` first; see Section 7 of the readme for metric definitions.
