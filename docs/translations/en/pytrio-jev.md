# Chapter 9.5: Train Your Own Jev Decision Model for the Price of Breakfast

<div align="center">
  <img src="./images/head.png" alt="Train your own Jev decision model for the price of breakfast" width="100%">
</div>

> **Code and reproduction resources**
>
> - Complete code: [`experiments/pytrio-jev`](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/experiments/pytrio-jev) / Quick start: [`start.md`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/pytrio-jev/start.md)
> - Method references: [Intern-Decision](https://github.com/InternLM/Intern-Decision), [Jev's architecture unmasked](https://archerhume.com/posts/jevs-architecture-unmasked), [Datawhale AgentJev tutorial](https://mp.weixin.qq.com/s/F7i83rn8oDzQ-sg13MKwWA)
> - Training data: [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), [`Team-ACE/ToolACE`](https://huggingface.co/datasets/Team-ACE/ToolACE)
> - SwanLab: [Public training logs for this experiment](https://swanlab.cn/@kmno4/agentic-rl-lab-pytrio-jev/v1/qt5fzd/runs/g4i73a65/chart)
> - PyTRIO: [Website](https://pytrio.com) / [Documentation](https://docs.pytrio.com/docs)

We trained on 2,160 records for two epochs, completing 270 LoRA steps. Training and evaluation cost RMB 14.34 in total, about the price of a McDonald's breakfast. On the same evaluation sets used by Intern-Decision, the five-benchmark average rose from the base model's 76.30 to 83.74. Our model scored above Jev on Typed Decision and ToolACE, and came close on most of the other benchmarks.

## 0. What exactly is Jev?

Online discussions can make models like Jev sound mysterious. Their basic idea is quite straightforward: **a classification and decision model with general world knowledge**. Give it a state and a set of questions with fixed options. Its interface asks for a choice and a confidence distribution: "A or B, and how confident are you in each?"

That modest interface is useful. Many business processes already have a small, fixed set of possible outcomes:

- **Intent recognition**: an after-sales request might concern a refund, a return with a refund, or another predefined resolution.
- **Ticket routing**: send this message to billing, logistics, or technical support.
- **Risk grading**: allow low-risk cases, review medium-risk cases, and block high-risk cases.

There is real spending behind these tasks. In the OpenRouter "Top models by task" snapshot below, **Classification accounts for 10.4% of platform spending**, the largest individual task within the General category, which accounts for 31.2%:

![OpenRouter Top models by task: Classification accounts for 10.4% of platform spending](./images/openrouter.png)

The models on that list are also interesting. Classification spending is led by expensive models such as Claude Opus 5, GPT-6 Astra, and Gemini 3.1 Pro. Businesses and users are paying for large general-purpose models to make classification decisions. A decision model that costs orders of magnitude less could handle this kind of demand.

These workflows need **a fast, stable decision with useful probabilities**. Having a general-purpose model deliberate at length for every three-way choice puts slow thinking to work on a fast-thinking task. I see a role for both: slow thinking handles open-ended reasoning and planning, while fast thinking handles frequent routing and decisions over enumerable options. Models like Jev fill that fast-thinking role.

How much does it take to build one? This experiment gives a concrete answer: you can train a decision model for your own domain for less than the price of a coffee.

## 1. Results first

We use exactly the evaluation sets in the official Intern-Decision leaderboard: three Jevbench difficulty levels, the Typed Decision test set, and 310 ToolACE examples. All values are accuracy percentages:

| Model | Easy | Original | Hard | Typed | ToolACE | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B (base) | 100.00 | 87.50 | 55.86 | 50.40 | 87.74 | 76.30 |
| **PyTrio-Jev (ours)** | 100.00 | 95.83 | 47.75 | 79.20 | 94.52 | **83.46** |
| **PyTrio-Jev (+ temperature calibration)** | 100.00 | 95.83 | 49.55 | 79.10 | 94.19 | **83.74** |
| Intern-Decision-0.8B | 97.92 | 80.56 | 52.25 | 77.35 | 94.52 | 80.52 |
| Intern-Decision-2B | 100.00 | 84.72 | 63.96 | 79.35 | 96.45 | 84.90 |
| Intern-Decision-4B | 100.00 | 98.61 | 73.87 | 80.55 | 96.45 | 89.90 |
| Jev | 100.00 | 98.61 | 72.07 | 73.35 | 91.29 | 87.06 |
| Laya | 95.83 | 72.22 | 28.83 | 35.95 | 63.87 | 59.34 |

Reference scores come from the [public Intern-Decision leaderboard](https://github.com/InternLM/Intern-Decision). Two differences matter when reading this comparison. The official models use **zero-shot** evaluation on Typed Decision; ours uses a specialist setup with the same workflows in training and testing. The official training uses full fine-tuning, while we use LoRA with only 2,400 general-purpose examples. For a business training on its own domain, I would expect more relevant data to help.

![Comparison across five benchmarks](./images/benchmark-comparison.png)

## 2. Method: one forward pass, one token

Here is the full training and inference pipeline:

![Training and inference pipeline](./images/pipeline.png)

### 2.1 Predict the answer at a marker

Ordinary SFT teaches a model to write an answer token by token. Here, the assistant-side input is a **JSON skeleton containing placeholders**, with one marker token for each question field. Supervision is applied only at the position immediately before each marker, where the model should predict the symbol for the correct answer: A/B/C, Yes/No, or 1–5.

The figure below shows a real training input after rendering the chat template. Highlighted positions contribute to the loss. To keep extra system-prompt text out of the decision input, we wrote a minimal [chat template](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/pytrio-jev/templates/qwen3_5_jev.jinja) containing only the conversation structure:

![Training input with marker supervision positions highlighted](./images/train-chat-template.png)

Three design choices matter:

- **Shared state, one forward pass**: each record becomes one sequence, and all fields receive supervision in the same forward pass. Field answers are absent from the input, so fields cannot see one another's answers. The question ordering also leaves the output probabilities unchanged.
- **One token locates each marker**: every placeholder is the single token `decision` (ID 61781). It identifies the supervision position during training and the readout position during inference. An ordinary token already in the vocabulary is sufficient.
- **Datum construction**: with the `cross_entropy` loss, `target_tokens` and nonzero `weights` are assigned only immediately before markers. All other positions have zero weight. We reuse the base model's representations and focus the training signal on which option to choose at these positions.

### 2.2 Inference: a restricted softmax over candidate symbols

Inference reads candidate-symbol log probabilities at the marker positions using `sample(max_tokens=1, topk_prompt_logprobs=K)`. It then applies a softmax over those candidates alone, with calibration temperature T:

![Restricted-softmax inference](./images/inference-principle.png)

All three question types use this mechanism, with different output mappings:

| Question type | Candidate symbols | Output |
| --- | --- | --- |
| `choice` (select one) | A/B/C/… | Argmax option and probability distribution |
| `noul` (yes/no judgment) | Yes/No | p(Yes) |
| `score` (rating) | 1/2/3/4/5 | Expected score and probability distribution |

The normalized probability distribution can feed threshold decisions, staged approvals, intent recognition, and ticket routing directly.

To get straight to the experiment, these four steps take you from cloning the repository to a full training run. Python 3.13 or newer is required:

```bash
# 0. 克隆仓库 + 安装依赖 + 登录
git clone https://github.com/KMnO4-zx/agentic-rl-lab.git
cd agentic-rl-lab
uv sync
trio login       # PyTRIO 训练服务（必需）
swanlab login    # SwanLab 训练曲线（可选，也可用 --swanlab-mode offline）

# 1. 下载并切分全部数据（训练 2160 条 + 五项评测集）
uv run python experiments/pytrio-jev/00-prepare-data.py

# 2. 冒烟：64 条跑 1 个 epoch，确认 loss 下降、marker 对齐正常
uv run python experiments/pytrio-jev/train.py --max-samples 64 --epochs 1 --save-every 0 --swanlab-mode offline

# 3. 正式训练：2160 条 × 2 epoch ÷ 16 = 270 步
uv run python experiments/pytrio-jev/train.py --epochs 2 --batch-size 16 --save-every 60 --swanlab-mode online
```

Training prints a weights path beginning with `trio://`. Next, fit the calibration temperature with `calibrate.py` (Section 5), then run the five benchmarks with `eval.py`. The complete inference, calibration, and evaluation commands are in [`start.md`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/pytrio-jev/start.md).

## 3. Data: 2,160 training records and the same evaluation sets as Intern-Decision

### 3.1 Training data

| Dataset | Size | Details |
| --- | --- | --- |
| `LocalLLaMA/typed-decisions` train split | 960 cases (4,800 questions) | Four workflows with 300 cases each, split at the case level into 240/30/30 for training/development/calibration; covers choice, score, and noul |
| `Team-ACE/ToolACE` | 1,200 records | **Remove the 310 evaluation rows before sampling**; convert each example into a choice question asking which tool to use for a request and tool list |

The total is **2,160 records and 6,000 decision questions**. Training supervision for noul comes only from typed-decisions, since ToolACE contains only choice questions. We will see the effect of that imbalance later.

### 3.2 Evaluation data: identical to Intern-Decision

| Benchmark | Size | Intern-Decision-4B reference |
| --- | ---: | ---: |
| Jevbench Easy | 48 | 100.00% |
| Jevbench Original | 72 | 98.61% |
| Jevbench Hard | 111 | 73.87% |
| Typed Decision test | 400 cases / 2,000 questions | 80.55% |
| ToolACE | 310 | 96.45% |

### 3.3 Preventing contamination

- The official typed-decisions training and test sets were generated independently, with different seeds and a `tr_` prefix on training IDs. We train on train and evaluate on test.
- The 310 ToolACE evaluation rows and all 231 Jevbench examples **are excluded from training**. We also check state-text hashes for duplicates across splits.
- Temperature is fitted only on the calibration split. The test sets are read after the checkpoint and T are fixed. The 120 cases in `dev.jsonl` are reserved for future checkpoint selection.

One command downloads and splits all the data:

```bash
uv run python 00-prepare-data.py
```

## 4. Training: 270 steps for the price of breakfast

The main configuration:

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen3.5-4B` |
| Training method | LoRA rank 32 (mlp/attn/unembed); the reference implementation uses full fine-tuning |
| Data | 2,160 records × 2 epochs |
| Batch size | 16 (about 45 supervised tokens per step) |
| Learning rate | `1e-4` |
| Total steps | 270 (weights saved automatically at the end of each epoch) |
| Framework | PyTRIO 0.2.9, fully asynchronous API |

The training loop is asynchronous. Calls to `forward_backward_async()` and `optim_step_async()` are submitted without blocking for their results, while a separate task consumes the logs. The 270 steps finish quickly.

```bash
# 冒烟（确认 loss 下降、marker 对齐正常）
uv run python train.py --max-samples 64 --epochs 1 --save-every 0 --swanlab-mode disabled

# 正式训练
uv run python train.py --epochs 2 --batch-size 16 --save-every 60 --swanlab-mode online
```

![SwanLab training curves](./images/swanlab-loss.png)

*The complete interactive curves are in the [SwanLab training logs](https://swanlab.cn/@kmno4/agentic-rl-lab-pytrio-jev/v1/qt5fzd/runs/g4i73a65/chart).*

Loss starts at 3.78 and drops below 1 within the first dozen or so steps. Fast convergence is unsurprising when each step supervises classification on only a few dozen tokens. The `supervised_tokens` curve fluctuates between 20 and 70 because records contain different numbers of fields.

![SwanLab run overview](./images/swanlab-overview.png)

**The full training run cost RMB 12.36 on PyTRIO, less than USD 2.**

![PyTRIO session costs](./images/pytrio-comsume.png)

Training consumed 2.72M training tokens and cost RMB 12.36. The two RMB 0.99 sessions above it are five-benchmark evaluation runs, each with 0.66M prefill tokens. Evaluation mainly reads long prompts in a forward pass and generates almost no text. Training and evaluation together came to less than RMB 15.

## 5. Calibration: make confidence meaningful

A decision model returns probabilities that downstream systems use for threshold decisions. When it reports 80% confidence, we want predictions at that confidence level to be correct about 80% of the time. Two common measures help check this:

- **ECE (Expected Calibration Error)**: group predictions into confidence bins, then compare each bin's confidence with its actual accuracy. Lower is better.
- **Brier score**: the mean squared error between predicted probabilities and the true labels. It penalizes incorrect answers, especially confident incorrect answers. Lower is better.

We use temperature scaling. On the calibration split of 120 cases and 600 questions, a binary search over inverse temperature minimizes NLL. The fitted value is **T = 1.13**, and NLL falls from 0.4990 to 0.4959:

```bash
uv run python calibrate.py --weights trio://...
```

![Before and after temperature calibration](./images/calibration-before-after.png)

The clearest improvement is on Jevbench Hard: ECE drops from 0.2311 to 0.1884, and Brier score drops from 0.6510 to 0.6331. The model was systematically overconfident on difficult questions; temperature scaling softens those probabilities. Mathematically, temperature leaves argmax unchanged. The small accuracy differences of roughly 0.1–2 percentage points between runs come from numerical variability in server-side sampling. Calibration improves the reliability of the probabilities themselves.

## 6. Why did training hurt the Hard benchmark?

Jevbench Hard accuracy falls from the base model's 55.86% to 47.75%. Training made this benchmark worse. **The training distribution and the Hard distribution differ substantially**:

- Training mostly contains short cases, with a median state length of about 727 characters. ToolACE often allows surface matching between a keyword in the request and a tool name.
- Hard contains long policy documents, with a median state length of about 1,611 characters, multi-hop reasoning (`multi_hop`), and difficult judgments involving several rules (`judge_hard`).

The model learns to spot keywords and make a quick decision on short cases. Applied to long policies, that habit produces premature, overconfident conclusions, consistent with the Hard ECE results. Our next priority is to expand training data toward the Hard distribution: longer documents, multi-hop questions, and conflicting rules.

## 7. File structure

```text
experiments/pytrio-jev/
├── 00-prepare-data.py    # 下载 typed-decisions / ToolACE / Jevbench，完成切分与防污染
├── templates/
│   └── qwen3_5_jev.jinja # 钉死的 chat template，训练/推理共用
├── data.py               # record → messages → tokens/weights，记录 marker 索引
├── train.py              # 全异步 LoRA SFT：cross_entropy + SwanLab，epoch 末存权重
├── inference.py          # topk_prompt_logprobs 受限 softmax 推理（choice/noul/score）
├── calibrate.py          # calibration 划分上拟合温度 T
├── eval.py               # 五项基准并发评测（acc / ECE / Brier）
├── analysis.py           # 汇总评测结果，生成对比图
├── start.md              # 从零开始的命令清单
└── images/
```

The complete command list is in [`start.md`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/pytrio-jev/start.md).

## Summary

The practical work in a Jev-style model is making classification decisions in one forward pass: a shared state, marker placeholders, a restricted softmax, and temperature calibration. This experiment used **2,160 training records and less than USD 2 of training cost**. It reached an average of 83.74 on the same evaluation sets as the official leaderboard, scoring above Jev on two tasks and close on most others, subject to the evaluation differences discussed above.

If your business has decisions with only a few possible options—intent recognition, ticket routing, or risk grading—this is a small experiment you can try with your own data. Open-ended reasoning and planning still need slow thinking; frequent, fixed-option decisions are a useful place to apply a fast-thinking model.

If you have a specific dataset and use case, feel free to contact me through Zhihu, Rednote, or email. I can help assess the feasibility and cost of training a Jev-style decision model. Email: violin@pytrio.com
