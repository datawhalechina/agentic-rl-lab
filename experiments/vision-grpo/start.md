# GeoQA Vision GRPO 快速启动

> 完整原理、训练结果与实现边界请查看 [Vision GRPO Blog](./readme.md)。

以下命令均从仓库根目录运行。项目要求 Python 3.13 及以上，训练与评测会调用 PyTRIO 远端服务。

> **成本提示**
>
> `train.py` 和 `eval.py` 会产生远端采样或训练消耗。先运行 1-step 小配置，确认图像、token 对齐与账号状态后，再启动完整实验。

## 1. 安装依赖并登录

```bash
uv sync
trio login
swanlab login
```

训练默认使用在线 SwanLab。调试时可传 `--swanlab-mode disabled`。

## 2. 下载并拆分 GeoQA

```bash
uv run python experiments/vision-grpo/download-dataset.py
```

默认生成：

```text
experiments/vision-grpo/datasets/train.parquet  # 3,503 条
experiments/vision-grpo/datasets/test.parquet   # 固定 100 条
```

需要锁定数据集 revision 时：

```bash
uv run python experiments/vision-grpo/download-dataset.py \
  --revision DATASET_COMMIT
```

## 3. 检查命令与本地导入

```bash
uv run python experiments/vision-grpo/download-dataset.py --help
uv run python experiments/vision-grpo/train.py --help
uv run python experiments/vision-grpo/eval.py --help
uv run python experiments/vision-grpo/analysis.py --help
```

## 4. 运行 1-step 小规模验证

```bash
uv run python experiments/vision-grpo/train.py \
  --steps 1 \
  --batch-size 1 \
  --group-size 4 \
  --max-tokens 256 \
  --save-every 0 \
  --no-save-weights \
  --show-samples \
  --swanlab-mode disabled
```

这个配置会检查真实图片采样、`input_tokens` 对齐、boxed reward 和 group advantage。单题四条回答可能全部同分，出现这种情况时该组会被跳过，日志中的 `train_datums` 为 0。

## 5. 启动 20-step 小规模训练

```bash
uv run python experiments/vision-grpo/train.py \
  --steps 20 \
  --batch-size 8 \
  --group-size 8 \
  --max-tokens 1024 \
  --learning-rate 4e-5 \
  --save-every 10 \
  --swanlab-mode online
```

## 6. 启动 100-step 正式训练

```bash
uv run python experiments/vision-grpo/train.py \
  --base-model Qwen/Qwen3.5-4B \
  --lora-rank 32 \
  --steps 100 \
  --batch-size 8 \
  --group-size 8 \
  --max-tokens 1024 \
  --temperature 1.0 \
  --top-p 1.0 \
  --learning-rate 4e-5 \
  --save-every 25 \
  --experiment-name vision-grpo-qwen35-4b-geoqa \
  --weights-name vision-grpo-qwen35-4b-geoqa \
  --swanlab-mode online
```

每 25 steps 会保存：

```text
trio://.../sampler_weights/...-sampler
trio://.../training_state/...-state
```

评测时使用 `sampler_weights`，恢复训练时使用 `training_state`。

## 7. 评测 Base

```bash
uv run python experiments/vision-grpo/eval.py \
  --base-model Qwen/Qwen3.5-4B \
  --max-tokens 1024 \
  --limit 100 \
  --output experiments/vision-grpo/eval-results-base.json
```

## 8. 评测 checkpoint

把 `--model-path` 替换为训练日志打印的 sampler weights：

```bash
uv run python experiments/vision-grpo/eval.py \
  --base-model Qwen/Qwen3.5-4B \
  --model-path 'trio://RUN_ID/sampler_weights/WEIGHTS_NAME' \
  --max-tokens 1024 \
  --limit 100 \
  --output experiments/vision-grpo/eval-results-step-100.json
```

每次评测一个模型，并为 Base 与 checkpoint 使用不同输出文件。

## 9. 生成文档中的结果图

```bash
uv run python experiments/vision-grpo/analysis.py
```

输出位置：

```text
experiments/vision-grpo/images/eval-comparison.png
```

当前 `analysis.py` 绘制的是文档记录的 Base `71.0% / 75.0%` 与 step 100 `87.0% / 91.0%`。新实验完成后，先更新 `BASE_RESULTS` 和 `GRPO_RESULTS`。
