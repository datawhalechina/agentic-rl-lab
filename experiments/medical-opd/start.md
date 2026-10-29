# OPD 快速启动

以下命令均从仓库根目录运行。项目要求 Python `>=3.13`。

```bash
uv sync
trio login
swanlab login
```

评测脚本通过 TRIO 的 OpenAI-compatible API 访问模型。尚未创建配置文件时执行：

```bash
cp experiments/medical-opd/.env.example experiments/medical-opd/.env
```

随后在 `experiments/medical-opd/.env` 中填写 `PYTRIO_API_KEY`。

## 1. 下载数据

```bash
uv run python experiments/medical-opd/00-download-dataset.py
```

这条命令会准备 Medical SFT、MedQA-zh 和 C-Eval 数据。

## 2. 启动训练

先训练 Medical SFT Teacher：

```bash
uv run python experiments/medical-opd/02-medical-sft.py \
    --num-epochs 3 \
    --batch-size 16 \
    --max-length 2048 \
    --swanlab-mode online
```

记录训练结束时打印的 SFT `sampler_weights` 路径，再启动 Medical OPD：

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

把示例中的 `RUN_ID` 和 `SFT_WEIGHTS_NAME` 替换为上一步实际输出。

## 3. 运行评测

把下面的模型路径替换为 OPD 训练日志打印的 sampler weights：

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

逐题结果与汇总指标会写入 `experiments/medical-opd/eval-results/`。
