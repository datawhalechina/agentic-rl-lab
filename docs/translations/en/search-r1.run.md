# Search-R1 Quick Start

Run all commands from the repository root. The project requires Python `>=3.13`.

```bash
uv sync
trio login
swanlab login
```

## 1. Download the data

```bash
uv run python experiments/search-r1/prepare_data.py
```

The default backend is the official DeepSeek Search API. Set the key in `experiments/search-r1/.env`:

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
```

`DEEPSEEK_KEY` is also supported. No installation or login for the legacy `deepseek-search` tool is required. Each request returns the top 3 results by default, passing short `snippet` excerpts, titles, sources and URLs to the model without extra model or Fetch API calls.

Wikipedia search does not require an API key. The client uses the English Wikimedia Action API with a default concurrency of 3, following its published limits.

## 2. Start training

DeepSeek Search backend:

```bash
uv run python experiments/search-r1/train.py \
    --max-steps 20 \
    --questions-per-batch 8 \
    --group-size 8 \
    --save-every 5 \
    --base-model Qwen/Qwen3.5-4B \
    --search-backend deepseek \
    --run-name search-r1-qwen35-4b-deepseek \
    --swanlab-mode online
```

Wikipedia backend:

```bash
uv run python experiments/search-r1/train.py \
    --max-steps 20 \
    --questions-per-batch 8 \
    --group-size 8 \
    --save-every 5 \
    --base-model Qwen/Qwen3.5-4B \
    --search-backend wikipedia \
    --run-name search-r1-qwen35-4b-wikipedia \
    --swanlab-mode online
```

For the full training run, set `--max-steps` to `100` and `--save-every` to `50`. Each step submits the entire rollout batch in one `forward_backward` call. The PyTRIO SDK splits requests and accumulates gradients automatically, followed by one `optim_step`. The per-trajectory training limit remains `8,192` tokens.

## 3. Run evaluation

The examples below use DeepSeek Search. First evaluate the base model:

```bash
uv run python experiments/search-r1/eval.py \
    --batch-size 16 \
    --base-model Qwen/Qwen3.5-4B \
    --search-backend deepseek \
    --search-count 3 \
    --search-concurrency 16 \
    --search-timeout 60 \
    --output experiments/search-r1/eval_result/eval_results_base_deepseek_api.jsonl
```

Then evaluate the trained sampler weights:

```bash
uv run python experiments/search-r1/eval.py \
    --batch-size 16 \
    --model-path 'trio://RUN_ID/sampler_weights/STEP_20_WEIGHTS_NAME' \
    --search-backend deepseek \
    --search-count 3 \
    --search-concurrency 16 \
    --search-timeout 60 \
    --output experiments/search-r1/eval_result/eval_results_rl_step_20_deepseek_api.jsonl
```

For Wikipedia, use `--search-backend wikipedia --search-concurrency 3 --search-timeout 15` in both commands. Base and checkpoint must use identical backends, excerpt fields and search settings. Historical Evidence results and new official Search API results come from different environments and must not be combined as a single comparison.
