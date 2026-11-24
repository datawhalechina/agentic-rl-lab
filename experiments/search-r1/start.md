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

可选的 Keenable 后端（`--search-backend keenable`）搜索全网，返回 Top N 网页的标题、`snippet` 片段（截断到 1200 字符）、来源域名和 URL。发送的只有搜索 query 和 `X-Keenable-Title: agentic-rl-lab` 请求头。请注意：无论是否设置 Key，请求都会按 Keenable 默认策略记录日志，只有企业版零数据保留（ZDR）不同，详见 [数据隐私说明](https://docs.keenable.ai/data-privacy)。

- 不设置 `KEENABLE_API_KEY` 时走免密钥接口，限额为每个 IP 10 次/秒、1000 次/小时（[限额说明](https://docs.keenable.ai/rate-limits.md)）。固定 70 题评测最多约 280 次搜索，够用；训练每个 step 最多约 256 次搜索，几个 step 就会用完小时额度，`train.py` 会打印警告。
- 完整训练请在 `experiments/search-r1/.env` 中设置 `KEENABLE_API_KEY`。
- 客户端把所有并发请求的启动速率限制在约 9 次/秒；429 按 `Retry-After` 退避重试，`Retry-After` 超过 30 秒时直接把错误返回给模型，不卡住 rollout。

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

评测 Wikipedia 后端时，两条命令都改为 `--search-backend wikipedia --search-concurrency 3 --search-timeout 15`；评测 Keenable 后端时改为 `--search-backend keenable --search-concurrency 8 --search-timeout 30`，summary 会记录实际使用的接口（免密钥或带 Key）和 `search_count`。Base 与 checkpoint 必须使用相同的后端、摘要字段和搜索配置。旧 Evidence 后端的历史结果不能与官方 Search API 的新结果混作同一环境比较。
