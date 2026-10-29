# 启动命令

以下命令都在 `experiments/pytrio-jev/` 目录内运行（`cd experiments/pytrio-jev` 之后）：

```bash
# 1. 下载与切分数据（typed-decisions train + ToolACE 1200 + 五项评测集）
uv run python 00-prepare-data.py

# 2. 小规模冒烟（确认 loss 下降、marker 对齐正常）
uv run python train.py --max-samples 64 --epochs 1 --save-every 0 --swanlab-mode offline

# 3. 正式训练（约 2160 条 × 2 epoch ÷ 16 = 270 步）
uv run python train.py --epochs 2 --batch-size 16 --save-every 60 --swanlab-mode online

# 4. 内置样例试跑推理（不传 --weights 则评 base 模型）
uv run python inference.py --weights trio://...

# 5. 温度校准（在 calibration 划分上拟合 T）
uv run python calibrate.py --weights trio://...

# 6. 五项基准评测（先校准，后读 test）
uv run python eval.py --weights trio://... --calibration outputs/calibration.json --tag step-270

# 7. 汇总对照参考基线
uv run python analysis.py
```
