# Tournament-GRPO 快速启动

以下命令都在仓库根目录运行，Python 版本 `>=3.13`。算法讲解见 [readme.md](./readme.md)。

## 0. 安装并登录

```bash
uv sync
uv run trio login
uv run swanlab login
```

## 1. 准备数据

```bash
uv run python experiments/tournament-grpo/prepare_data.py
```

脚本下载作者固定 commit 的训练数据和 Deep Research Bench，并用 `Qwen/Qwen3.5-4B` tokenizer 去掉超过 4096 token 的题目：

```text
experiments/tournament-grpo/datasets/
├── raw/                      # 作者原始 Parquet
├── train.jsonl               # 8,378 条
├── val.jsonl                 # 440 条
└── deep_research_bench/      # query / criteria / reference 各 100 条
```

## 2. 配置 API Key

```bash
cp -n experiments/tournament-grpo/.env.example experiments/tournament-grpo/.env
```

在 `experiments/tournament-grpo/.env` 中填写搜索、训练裁判和评测裁判：

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

- `JINA_API_KEY` 可以留空，此时限速 20 RPM。国内可把 `JINA_READER_BASE_URL` 改成 `https://r.jinaai.cn`。
- Jev 官方接口在中国大陆无法直接访问。工具和裁判请求不读取 shell 的 `HTTP(S)_PROXY`，国内读者需要开启系统级全局代理，或把 `TOURNAMENT_JUDGE_API_URL` 换成国内可访问的兼容服务。
- 搜索和 Jina 也可以在 `.env` 中分别设置 `DEEPSEEK_SEARCH_PROXY`、`JINA_READER_PROXY`。

检查 Jev 是否可用：

```bash
uv run python experiments/tournament-grpo/judge.py
```

## 3. 单步验收

```bash
uv run python experiments/tournament-grpo/train.py \
  --max-steps 1 \
  --questions-per-batch 1 \
  --group-size 8 \
  --run-name tournament-grpo-smoke
```

这一步采样 1 道题、8 条轨迹，进行 28 场裁判比较，并更新一次权重，约需 2–5 分钟。运行后检查 `experiments/tournament-grpo/outputs/tournament-grpo-smoke/train.jsonl`：`judge/comparisons` 为 28，`reward/has_answer` 大于 0。

## 4. 20-step 训练

```bash
uv run python experiments/tournament-grpo/train.py \
  --max-steps 20 \
  --questions-per-batch 8 \
  --group-size 8 \
  --learning-rate 4e-5 \
  --save-every 10 \
  --run-name tournament-grpo-20step
```

每步 64 条轨迹、224 场裁判比较。裁判请求重试 3 次仍失败时，训练会停止。修好接口后，从最近的 checkpoint 续训：

```bash
uv run python experiments/tournament-grpo/train.py \
  --max-steps 20 \
  --resume experiments/tournament-grpo/outputs/tournament-grpo-20step/checkpoint-step-10.json \
  --run-name tournament-grpo-20step-resumed
```

训练输出位于 `experiments/tournament-grpo/outputs/<run-name>/`：

```text
train.jsonl                 # 每步指标
judge.jsonl                 # 每场比赛的候选、胜者和裁判输出
rollout-step-*.jsonl        # 第 1 步和 checkpoint 步的完整轨迹
checkpoint-step-*.json      # 续训 state 和采样权重
```

## 5. Deep Research Bench 评测

先为 Base 和 Step 20 生成研究报告：

```bash
uv run python experiments/tournament-grpo/eval.py generate \
  --batch-size 16
uv run python experiments/tournament-grpo/eval.py generate \
  --batch-size 16 \
  --checkpoint experiments/tournament-grpo/outputs/tournament-grpo-20step/checkpoint-step-20.json
```

再用 RACE 打分：

```bash
uv run python experiments/tournament-grpo/eval.py score \
  --judge-concurrency 8
uv run python experiments/tournament-grpo/eval.py score \
  --judge-concurrency 8 \
  --checkpoint experiments/tournament-grpo/outputs/tournament-grpo-20step/checkpoint-step-20.json
```

评测使用 100 道题，每题贪心生成 1 条轨迹。生成和评分都逐题保存，中断后重跑同一命令即可继续。可以先加 `--limit 4`，只跑前 4 题检查流程。

结果按 policy 模型、权重和裁判模型分开保存：

```text
experiments/tournament-grpo/outputs/drb/Qwen--Qwen3.5-4B/
├── base/
│   ├── articles.jsonl
│   ├── generation-summary.json
│   └── race/deepseek-flash/race-summary.json
└── trio--run_<ID>--sampler_weights--step-20-weights/
    └── ...
```

## 6. 查看结果

训练曲线在 SwanLab 的 `tournament-grpo` 项目中查看。优先看 `reward/*` 和 `rollout/*`，指标含义见 readme 第 7 节。
