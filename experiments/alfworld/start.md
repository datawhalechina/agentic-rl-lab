# ALFWorld Agentic RL 快速启动

> 完整原理、训练结果与代码拆解请查看 [ALFWorld Agentic RL Blog](./readme.md)。

以下命令均从仓库根目录运行。项目要求 Python `>=3.13`。

> **成本提示**
>
> 正式配置每个 update 采样 64 条长轨迹，单条序列最多 12K tokens。本文 80-step 实验的 PyTRIO 账单为 ¥1613.71。建议先运行 1-step 验证，再启动完整训练。

## 1. 安装依赖并登录

ALFWorld 使用可选依赖安装，不会影响仓库中的其他子项目：

```bash
uv sync --extra alfworld
trio login
swanlab login
```

## 2. 下载 ALFWorld 数据

```bash
uv run --extra alfworld alfworld-download \
    --data-dir "$PWD/experiments/alfworld/datasets/alfworld"
```

训练和评测默认读取该目录，无需额外传入 `--data-root`。

## 3. 运行 1-step 验证

先用一个游戏、8 条轨迹检查数据、环境、采样、reward、PPO 和 checkpoint 链路：

```bash
uv run --extra alfworld python experiments/alfworld/train.py \
    --max-steps 1 \
    --games-per-batch 1 \
    --group-size 8 \
    --max-episode-steps 10 \
    --save-every 0 \
    --run-name alfworld-smoke \
    --swanlab-mode disabled
```

`--save-every 0` 只关闭中间 checkpoint；训练结束后仍会保存最终 state 和 sampler weights。

## 4. 启动 80-step 正式训练

```bash
uv run --extra alfworld python experiments/alfworld/train.py \
    --base-model Qwen/Qwen3.5-4B \
    --max-steps 80 \
    --games-per-batch 8 \
    --group-size 8 \
    --max-episode-steps 50 \
    --max-trajectory-tokens 12000 \
    --max-assistant-tokens 2048 \
    --temperature 1.0 \
    --top-p 1.0 \
    --learning-rate 1e-6 \
    --save-every 20 \
    --run-name alfworld-agent-rl-qwen35-4b \
    --swanlab-mode online
```

每 20 updates 会同时打印两类路径：

```text
Saved state: trio://.../training_state/...
Saved sampler weights: trio://.../sampler_weights/...
```

评测 checkpoint 时使用 `Saved sampler weights`，不要使用 state 路径。

## 5. 评测 Base Model

完整评测覆盖 `valid_seen=140` 与 `valid_unseen=134`，一共 274 个游戏：

```bash
uv run --extra alfworld python experiments/alfworld/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/alfworld/eval_results/base-qwen35-4b-eval.jsonl \
    --swanlab-mode disabled
```

## 6. 评测 checkpoint

将 `--model-path` 替换为训练日志打印的 sampler weights 路径。下面以 Step 80 为例：

```bash
uv run --extra alfworld python experiments/alfworld/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path 'trio://RUN_ID/sampler_weights/WEIGHTS_NAME' \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/alfworld/eval_results/checkpoint-80steps.jsonl \
    --swanlab-mode disabled
```

`eval.py` 默认拒绝覆盖已有 JSONL。需要重新评测同一路径时，显式增加：

```bash
--overwrite-output
```

## 7. 生成评测图

`analysis.py` 默认读取以下三个文件：

```text
experiments/alfworld/eval_results/base-qwen35-4b-eval.jsonl
experiments/alfworld/eval_results/checkpoint-40steps.jsonl
experiments/alfworld/eval_results/checkpoint-80steps.jsonl
```

三个文件准备完成后运行：

```bash
uv run python experiments/alfworld/analysis.py
```

结果图会保存到：

```text
experiments/alfworld/images/alfworld_checkpoint_evaluation.png
```
