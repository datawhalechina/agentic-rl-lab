# Search-R1 快速启动

以下命令均从仓库根目录运行。项目要求 Python `>=3.13`。

```bash
uv sync
trio login
swanlab login
```

## 1. 下载数据

```bash
uv run python experiments/search-r1/prepare_data.py
```

默认使用 DeepSeek 官方 Search API。在 `experiments/search-r1/.env` 中配置：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
```

也兼容已有的 `DEEPSEEK_KEY`。无需安装或登录旧的 `deepseek-search` 工具。每次请求默认返回 Top 3，优先将 `snippet` 短摘要连同标题、来源、URL 交给模型；不额外调用模型或 Fetch API。

Wikipedia 搜索无需 API Key。客户端会使用英文 Wikimedia Action API，并按其公开限额默认使用 3 个并发。

## 2. 启动训练

DeepSeek Search 后端：

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

Wikipedia 后端：

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

正式训练可把 `--max-steps` 改为 `100`，并把 `--save-every` 改为 `50`。每个 step 将完整 rollout batch 一次提交给 `forward_backward`，由 PyTRIO SDK 自动拆批和累积梯度，然后执行一次 `optim_step`。单条训练轨迹上限保留为 `8,192` tokens。

## 3. 运行评测

下面以 DeepSeek Search 为例。先评测 Base Model：

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

再评测训练得到的 sampler weights：

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

评测 Wikipedia 后端时，两条命令都改为 `--search-backend wikipedia --search-concurrency 3 --search-timeout 15`。Base 与 checkpoint 必须使用相同的后端、摘要字段和搜索配置。旧 Evidence 后端的历史结果不能与官方 Search API 的新结果混作同一环境比较。
