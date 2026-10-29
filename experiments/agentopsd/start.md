# AgentOPSD 快速启动

> 完整原理、训练结果与代码拆解请查看 [AgentOPSD Blog](./readme.md)。

以下命令均从仓库根目录运行。项目要求 Python `>=3.13`；本地负责 ALFWorld 环境、轨迹编排和 turn credit，模型采样与 LoRA 训练由 PyTRIO 远端执行。

> **成本提示**
>
> 正式配置每个 update 会采样 `16 × 8 = 128` 条长轨迹，80 steps 共 10,240 条 rollout，并额外执行同轨迹 Teacher 复评。本文这次正式训练的 PyTRIO 账单为 ¥675.69。建议先完成 1-step smoke，再启动完整训练。

## 1. 安装依赖并登录

```bash
uv sync --extra alfworld
trio login
swanlab login
```

`swanlab login` 仅在使用 `--swanlab-mode online` 时需要。检查当前关键依赖版本：

```bash
uv run --extra alfworld python -c 'from importlib.metadata import version; print("pytrio", version("pytrio")); print("alfworld", version("alfworld")); print("swanlab", version("swanlab"))'
```

本文锁定的版本为 PyTRIO `0.2.8`、ALFWorld `0.4.2` 和 SwanLab `0.9.2`。

## 2. 下载 ALFWorld 数据

```bash
uv run --extra alfworld alfworld-download \
    --data-dir "$PWD/experiments/agentopsd/datasets/alfworld"
```

训练和评测默认读取该目录，无需额外传入 `--data-root`。

## 3. 查看可用参数

```bash
uv run --extra alfworld python experiments/agentopsd/train.py --help
uv run --extra alfworld python experiments/agentopsd/eval.py --help
uv run python experiments/agentopsd/analysis.py --help
```

## 4. 运行 1-step smoke

先用一个游戏、4 条轨迹检查 ALFWorld 环境、Student rollout、同快照 Teacher 复评、turn credit、PPO 和 checkpoint 链路：

```bash
uv run --extra alfworld python experiments/agentopsd/train.py \
    --max-steps 1 \
    --tasks-per-update 1 \
    --group-size 4 \
    --max-turns 20 \
    --teacher-concurrency 4 \
    --save-every 1 \
    --seed 3 \
    --run-name agentopsd-smoke \
    --swanlab-mode disabled
```

单个游戏的 4 条轨迹可能全部成功或全部失败，此时 group-relative advantage 为 0，脚本会保留 rollout 指标并跳过参数更新。需要稳定验证 backward 时，可以提高 `--tasks-per-update`。

## 5. 启动 80-step 正式训练

```bash
uv run --extra alfworld python experiments/agentopsd/train.py \
    --base-model Qwen/Qwen3.5-4B \
    --max-steps 80 \
    --tasks-per-update 16 \
    --group-size 8 \
    --max-turns 50 \
    --max-trajectory-tokens 14336 \
    --max-action-tokens 512 \
    --temperature 1.0 \
    --top-p 1.0 \
    --teacher-concurrency 16 \
    --gamma 0.95 \
    --reshape-lambda 0.5 \
    --weight-bound 0.2 \
    --ppo-clip-low 0.8 \
    --ppo-clip-high 1.24 \
    --lora-rank 32 \
    --learning-rate 4e-6 \
    --save-every 40 \
    --seed 42 \
    --run-name agentopsd-alfworld-qwen35-4b \
    --swanlab-project agentic-rl-lab-agentopsd \
    --swanlab-mode online
```

Step 40、Step 80 和训练结束时都会打印两类路径：

```text
Saved state: trio://.../training_state/...
Saved sampler weights: trio://.../sampler_weights/...
```

`eval.py` 使用 `Saved sampler weights` 路径。

## 6. 快速检查评测链路

下面只评测 seen / unseen 各 2 个游戏，适合先检查输出路径和环境：

```bash
uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --split all \
    --limit-per-split 2 \
    --games-per-batch 2 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/base-smoke.jsonl \
    --swanlab-mode disabled
```

## 7. 完整评测 Base Model

完整评测覆盖 `valid_seen=140` 与 `valid_unseen=134`，一共 274 个游戏：

```bash
uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/base-qwen35-4b-eval.jsonl \
    --swanlab-mode disabled
```

## 8. 评测自己的 checkpoint

把 `--model-path` 替换为训练日志打印的 sampler weights 路径：

```bash
uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path 'trio://RUN_ID/sampler_weights/WEIGHTS_NAME' \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/checkpoint-eval.jsonl \
    --swanlab-mode disabled
```

`eval.py` 默认保护已有 JSONL。确认需要重跑同一输出文件时，显式增加：

```bash
--overwrite-output
```

## 9. 复现本文 Step 40 与 Step 80 评测

```bash
uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path 'trio://run_sschmbprfwg0/sampler_weights/agentopsd-alfworld-qwen35-4b-step-40-weights' \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/checkpoint-eval-step40.jsonl \
    --swanlab-mode disabled

uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path 'trio://run_sschmbprfwg0/sampler_weights/agentopsd-alfworld-qwen35-4b-step-80-weights' \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/checkpoint-eval-step80.jsonl \
    --swanlab-mode disabled
```

## 10. 生成 checkpoint 对比图

`analysis.py` 默认读取：

```text
experiments/agentopsd/eval_results/base-qwen35-4b-eval.jsonl
experiments/agentopsd/eval_results/checkpoint-eval-step40.jsonl
experiments/agentopsd/eval_results/checkpoint-eval-step80.jsonl
```

三个文件准备完成后运行：

```bash
uv run python experiments/agentopsd/analysis.py
```

结果图保存到：

```text
experiments/agentopsd/images/agentopsd_checkpoint_evaluation.png
```

自定义结果目录、输出位置或分辨率：

```bash
uv run python experiments/agentopsd/analysis.py \
    --result-dir experiments/agentopsd/eval_results \
    --output experiments/agentopsd/images/agentopsd_checkpoint_evaluation.png \
    --dpi 300
```
