# Reproducing Ten RL Algorithms, Chapter 7: GSPO from Scratch and a Six-Point AIME25 Gain

> Ready to run the code? See the [quick start](./start.md): prepare data, train, then evaluate.

![GSPO experiment overview](./images/封面.png)

<div align="center">
  <a href="https://www.zhihu.com/people/feng-qi-xia-pian" target="_blank"><img alt="Zhihu" src="https://img.shields.io/badge/Zhihu-知乎-4362f6"></a>
  <a href="https://www.xiaohongshu.com/user/profile/63c2055e000000002502c58c" target="_blank"><img alt="Rednote" src="https://img.shields.io/badge/Rednote-小红书-e93c49"></a>
  <a href="https://github.com/datawhalechina/agentic-rl-lab"><img alt="visitors" src="https://komarev.com/ghpvc/?username=KMnO4-zx-agentic-rl-lab-gspo&amp;label=visitors&amp;color=1283c3&amp;style=flat"></a>
</div>

> **Code and reproduction resources**
>
> - Complete code: [datawhalechina/agentic-rl-lab/experiments/gspo](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/gspo)
> - GSPO paper: [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
> - SwanLab: [Complete 100-step training record](https://swanlab.cn/@kmno4/llm-agent-rl-lab-gspo/runs/o83z2ghp/chart)
> - PyTRIO documentation: [https://docs.pytrio.com/docs](https://docs.pytrio.com/docs)

This is the seventh chapter in my series reproducing ten reinforcement learning algorithms.

For DAPO, we added Dynamic Sampling, Clip-Higher, Token Mean and Soft Overlong Punishment to the GRPO framework. GSPO makes a more focused change: **keep group rollouts, rewards and group-relative advantages, and move the importance ratio and clipping in the policy loss from individual tokens to complete sequences.**

Using [PyTRIO](https://pytrio.com/), I completed 100 steps of LoRA training on `Qwen/Qwen3.5-4B` and evaluated the base model plus five checkpoints at steps 20, 40, 60, 80 and 100. Let's examine the algorithm before the results.


## What is GSPO?

**Group Sequence Policy Optimization**, or GSPO, was introduced by the Qwen team in 2025. It samples a group of responses to each question and calculates advantages from their relative rewards, avoiding a separate value model. GSPO retains these group-relative advantages and changes the loss:

![](./images/GSPO vs GRPO · Sequence Ratio.png)

> **GRPO: a response with T tokens has T ratios and T clipping decisions. GSPO: a response has one ratio and one clipping decision.**

| Stage | GRPO | GSPO |
| --- | --- | --- |
| Group rollout | Sample $G$ responses to one prompt | Same |
| Reward | One reward per complete response | Same |
| Advantage | Normalize within the group; one $\hat A_i$ per response | Same |
| Importance ratio | One $w_{i,t}$ per token | One $s_i$ per response |
| Clipping | Decide whether to keep or clip each token | One decision for the complete response |
| Loss | Aggregate token objectives first | Aggregate sequence objectives directly |

GRPO calculates the new-to-old policy probability ratio separately for each token:

```math
w_{i,t}(\theta)=
\frac{
  \pi_\theta\left(y_{i,t}\mid x,y_{i,\lt t}\right)
}{
  \pi_{\theta_{\mathrm{old}}}\left(y_{i,t}\mid x,y_{i,\lt t}\right)
}
```

A response containing $T$ tokens produces $T$ ratios and $T$ clipping decisions.

GSPO first averages token log-ratios over the entire response, then exponentiates:

```math
s_i(\theta)=
\exp\left(
\frac{1}{|y_i|}
\sum_{t=1}^{|y_i|}
\log
\frac{
  \pi_\theta\left(y_{i,t}\mid x,y_{i,\lt t}\right)
}{
  \pi_{\theta_{\mathrm{old}}}\left(y_{i,t}\mid x,y_{i,\lt t}\right)
}
\right)
```

The factor $1/|y_i|$ normalizes for length, keeping ratios for different response lengths in comparable numerical ranges. Every token in the response then shares the same ratio and clipping result:

```math
\mathcal{J}_{\mathrm{GSPO}}(\theta)
=
\frac{1}{G}\sum_{i=1}^{G}
\min\left(
s_i\hat{A}_i,\,
\mathrm{clip}\left(
  s_i,\,
  1-\varepsilon_{\mathrm{low}},\,
  1+\varepsilon_{\mathrm{high}}
\right)\hat{A}_i
\right)
```

The GSPO paper uses these clipping bounds:

```text
epsilon_low  = 3e-4
epsilon_high = 4e-4
```

These bounds look much smaller than common GRPO bounds because they constrain different quantities. GRPO constrains each token's probability ratio, whereas GSPO constrains a length-normalized sequence ratio.

The paper argues that rewards are already assigned to complete responses. A sequence-level ratio aligns rewards, importance sampling and optimization at the same granularity, while reducing the high-variance noise of token-level ratios in long sequences.

This project's mathematical reward uses `math_verify` to check the final answer: `1` for correct and `0` for incorrect. Rewards within a group are normalized as follows:

```math
\hat{A}_i =
\frac{
  r_i-\mathrm{mean}(r)
}{
  \mathrm{std}(r)+10^{-8}
}
```

When all answers in a group are correct or all are incorrect, every advantage is 0. This implementation follows the basic GSPO objective without adding Dynamic Sampling or replacement questions. Training skips sequences with zero gradients but retains their zero objectives in the original loss denominator.

## Reproduction results

The full training run uses this configuration:

| Item | Configuration |
| --- | --- |
| Base Model | `Qwen/Qwen3.5-4B` |
| LoRA rank | 32 |
| Training data | DAPO-Math, with 17,126 training questions after cleaning |
| Training duration | 100 steps |
| Per step | 8 prompt groups |
| Per group | 8 completions |
| Rollout per step | 64 completions |
| Maximum prompt / completion length | 2,048 / 4,096 tokens |
| Sampling | temperature 1.0 / top-p 1.0 / top-k -1 |
| Optimizer | Adam，lr `4e-5`，β `(0.9, 0.95)` |
| Checkpoint | Save every 20 steps |

SwanLab records all 100 steps, averaging about 85 seconds per step and about 2 hours 22 minutes in total. The following figures show rollout/reward and GSPO loss metrics:

![](./images/swanlab-reward.png)

![](./images/swanlab-gspo.png)

During training, `normalization_sequences` stays at the original 64 responses per step, while `train_sequences` varies with the number of degenerate groups. The mean sequence clip fraction across 100 steps was 13.11%, and the final step was 6.25%. See the [SwanLab training record](https://swanlab.cn/@kmno4/llm-agent-rl-lab-gspo/runs/o83z2ghp/chart) for the full configuration and per-step curves.

The PyTRIO session for this 100-step run recorded a cost of `¥155.80`:

![](./images/pytrio-consume.png)

### AIME25 evaluation

The base model and every checkpoint use the same evaluation configuration:

```text
30 道 AIME25
每题采样 12 次，共 360 条 generations
temperature = 1.0
top_p = 1.0
top_k = -1
max_tokens = 8192
```

The metrics are:

- **Average@12**: the proportion of correct answers among 360 generations.
- **Pass@12**: the proportion of the 30 questions with at least one correct sampled answer.
- **Format**: the proportion of responses from which a `\boxed{}` answer was successfully extracted.

The complete results are:

| Checkpoint | Average@12 | Pass@12 | Format | Correct answers | Questions passed | Mean completion tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base (Run 1) | 41.11% | 56.67% | 46.67% | 148/360 | 17/30 | 6489.8 |
| Base (Run 2) | 40.56% | 66.67% | 47.50% | 146/360 | 20/30 | 6442.5 |
| Step 20 | 41.11% | 73.33% | 62.22% | 148/360 | 22/30 | 5775.4 |
| Step 40 | 37.78% | 60.00% | 87.22% | 136/360 | 18/30 | 4785.8 |
| Step 60 | 35.00% | 63.33% | 92.50% | 126/360 | 19/30 | 4438.7 |
| Step 80 | 34.72% | 73.33% | 97.78% | 125/360 | 22/30 | 3886.9 |
| Step 100 | 33.89% | 66.67% | 99.44% | 122/360 | 20/30 | 3358.1 |

![](./images/aime25-gspo-progress.png)

The cover's claim of a six-point AIME25 gain refers specifically to **Step 20 Pass@12 increasing from Base Run 2's 66.67% to 73.33%, a gain of 6.66 percentage points**.

Three other observations matter more:

1. Step 20 Average@12 is 41.11%, within the variation of the two base-model evaluations. Its number of correct answers also matches Base Run 1: 148/360.
2. The two independent base-model runs differ by 10 percentage points in Pass@12. With only 30 AIME25 questions, a single Pass@12 result is sensitive to sampling variation. The 6.66-percentage-point gain therefore needs more repeated evaluations and random seeds.
3. Format rises steadily from 47.50% to 99.44%, while mean completion length falls from 6442.5 to 3358.1 tokens, a reduction of about 48%. After Step 20, Average@12 gradually declines.

The best-supported conclusion is that **GSPO substantially changed output behavior: format compliance improved and responses became shorter. These data do not yet demonstrate a stable improvement in mathematical accuracy.** Different metrics also favor different checkpoints: Step 20 has the best Average@12, Steps 20 and 80 tie for the best Pass@12, and Step 100 has the best format rate.

A GRPO control with the same model, data and training budget is still missing, so these results do not establish that GSPO outperforms GRPO.

> **Reproduction boundaries**
>
> The original GSPO paper starts from a cold-start model fine-tuned from `Qwen3-30B-A3B-Base` and splits each rollout batch into 4 optimizer mini-batches. It reports AIME 2024, LiveCodeBench and CodeForces results. This chapter uses Qwen3.5-4B LoRA, one update per rollout batch and AIME25. It reproduces the core GSPO loss and training pipeline, not the paper-scale experiments or its MoE stability findings. The evaluation results should be interpreted within that scope.

## How to start training

The project requires Python `>=3.13`. Install dependencies and log in to PyTRIO and SwanLab:

```bash
git clone https://github.com/datawhalechina/agentic-rl-lab.git
cd agentic-rl-lab
uv sync
trio login
swanlab login
```

Prepare the DAPO-Math training data and AIME25:

```bash
uv run python experiments/gspo/prepare_data.py
```

Launch the same 100-step configuration used in this article:

```bash
cd experiments/gspo

uv run python train.py \
    --max-steps 100 \
    --groups-per-step 8 \
    --group-size 8 \
    --max-prompt-tokens 2048 \
    --max-tokens 4096
```

The PyTRIO service performs model sampling, LoRA forward and backward passes, optimizer updates and checkpoint storage. The local machine handles data, rewards, the GSPO loss and training control.

## How did we implement it?

This diagram outlines the pipeline using the current default of 4 prompt groups. The full experiment above overrides that default to 8 through the command line.

![](./images/GSPO × PyTRIO · Training Loop.png)

### 1. Prepare and pin the mathematical data

[`prepare_data.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/prepare_data.py) downloads a pinned revision of DAPO-Math-17K, cleans prompts, deduplicates examples and creates train/dev splits. It separately downloads and validates the 30 AIME25 questions.

[`data.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/data.py) reads training JSONL, shuffles with a fixed seed and uses a wrapping `ExampleCursor` to select questions per step. The current implementation has no Dynamic Sampling: it takes a fixed batch of prompts and samples them once.

### 2. Group rollouts, binary rewards and advantages

[`rollout.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/rollout.py) builds prompts with the Qwen chat template, then concurrently samples 8 responses per question using a PyTRIO sampler for the current LoRA weights. Each response retains:

```text
completion tokens
sampling logprobs
decoded text
reward
advantage
```

Token and sampling-logprob lengths are strictly checked because the sequence ratio requires token-by-token alignment.

[`reward.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/reward.py) extracts the final complete `\boxed{}` answer and uses `math_verify` to check equivalence with the reference answer. Correct answers receive 1; incorrect answers and missing answer formats receive 0. There is no length penalty or additional format reward.

After rollout, the 8 rewards for each question provide a group mean and sample standard deviation, producing a sequence advantage shared by every token in each response. All-correct and all-incorrect groups have zero advantage and are omitted from remote gradient computation.

### 3. Build the Datum and calculate the GSPO loss locally

[`loss.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/loss.py) is the core of this reproduction.

`build_datum()` applies the autoregressive shift to the prompt and completion:

```text
model_input   = prompt_tokens + completion_tokens[:-1]
target_tokens = [0] * (len(prompt_tokens) - 1) + completion_tokens
```

PyTRIO's `Datum.loss_fn_inputs` stores `target_tokens`. Local `GSPOMeta` stores rollout `sampling_logprobs`, the sequence advantage and completion length.

Next, `forward_backward_custom()` returns differentiable log probabilities for these target tokens under the current policy. `make_gspo_loss_fn()` selects the completion positions and performs three calculations:

```text
current logprobs - sampling logprobs
        ↓
mean token log-ratio → exp → one sequence ratio
        ↓
one sequence-level clipping decision
```

Sequences contribute with equal weight, and their sum is divided by the original rollout sequence count. Although degenerate groups are not sent for remote computation, their zero objectives remain represented in the denominator. Filtering therefore saves computation without amplifying the remaining samples' loss.

### 4. One update per rollout batch

[`train.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/train.py) connects the modules:

```text
加载并打乱数据
→ 创建 LoRA TrainingClient，并在训练开始时获取一次 tokenizer
→ 每 step 用最新 LoRA 权重创建 sampler
→ 并发 group rollout
→ 过滤 advantage 为 0 的退化组
→ forward_backward_custom
→ optim_step
→ 记录 SwanLab，并按周期保存 checkpoint
```

The application does not split the rollout into additional mini-batches. Each rollout batch makes one backward call and one `optim_step`. PyTRIO handles backend model computation, physical sharding and gradient accumulation.

### 5. Evaluate and plot the results

[`eval.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/eval.py) evaluates either the base model or any checkpoint identified by `trio://` sampler weights. It saves all 12 responses per question and aggregates Average@12, Pass@12, Format and mean completion length.

[`analyse.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/analyse.py) reads summaries from `eval-results/`, sorts them by checkpoint and generates the 1×2 AIME25 result figure used here.

The eight code files have these responsibilities:

| File | Responsibility |
| --- | --- |
| [`prepare_data.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/prepare_data.py) | Download, clean, deduplicate and pin DAPO-Math and AIME25 |
| [`data.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/data.py) | Read training questions, shuffle with a fixed seed and sample cyclically |
| [`reward.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/reward.py) | Extract `\boxed{}`, check mathematical equivalence and assign binary rewards |
| [`rollout.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/rollout.py) | Prompts, concurrent group rollouts, token/logprob alignment and advantages |
| [`loss.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/loss.py) | Datum construction, GSPO metadata, sequence ratios and clipped loss |
| [`train.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/train.py) | Main loop, PyTRIO calls, SwanLab and checkpoints |
| [`eval.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/eval.py) | Base/LoRA AIME25 evaluation and JSONL results |
| [`analyse.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/gspo/analyse.py) | Aggregate checkpoint metrics and plot results |

## Summary

GSPO retains GRPO's group rollouts, rewards and relative advantages while changing the unit of importance sampling and clipping from tokens to complete sequences. We implemented this custom-loss training pipeline with PyTRIO and evaluated the base model plus 5 checkpoints on AIME25. Step 20 improved Pass@12 by 6.66 percentage points in one comparison, while changes in format compliance and response length were more consistent. Establishing a sustained mathematical-accuracy gain still requires a matched GRPO control, more random seeds and repeated evaluations.
