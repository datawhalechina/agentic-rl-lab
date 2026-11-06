# GRPO 快速启动

以下命令均从仓库根目录运行。项目要求 Python `>=3.13`。

```bash
uv sync
trio login
swanlab login
```

## 1. 准备数据

无需单独执行下载脚本。训练首次启动时会自动下载 `openai/gsm8k` 的 train split，并使用本地 Hugging Face 缓存。

## 2. 启动训练

同步版示例：

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode online
```

`02-demo-async.py` 与同步版逻辑相同，只是把 batch 内每道题的 rollout 并发提交出去；它已适配当前 PyTRIO 0.2.9 的配置参数，可以直接运行。

## 3. 运行评测

当前项目没有独立的 `eval.py`。训练过程会在终端和 SwanLab 中记录 `reward`、`frac_degenerate`、`rollout/avg_gen_len`、`train_tokens` 与 `loss_mean`，并在结束时打印保存的 sampler weights 路径。

比较不同 loss 时，保持其他参数一致，仅修改：

```bash
--loss-fn importance_sampling
--loss-fn ppo
--loss-fn cispo
```
