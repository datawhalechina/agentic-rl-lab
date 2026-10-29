# Reproducing Ten RL Algorithms, Chapter 6: What DAPO Put Me Through

> Ready to run the code? See the [quick start](./start.md): prepare data → train → evaluate.

![](./images/head.png)

<div align="center">
  <a href="https://www.zhihu.com/people/feng-qi-xia-pian" target="_blank"><img alt="Zhihu" src="https://img.shields.io/badge/Zhihu-知乎-4362f6"></a>
  <a href="https://www.xiaohongshu.com/user/profile/63c2055e000000002502c58c" target="_blank"><img alt="Rednote" src="https://img.shields.io/badge/Rednote-小红书-e93c49"></a>
  <a href="https://github.com/KMnO4-zx/agentic-rl-lab"><img alt="visitors" src="https://komarev.com/ghpvc/?username=KMnO4-zx-agentic-rl-lab-dapo&amp;label=visitors&amp;color=1283c3&amp;style=flat"></a>
</div>

> **Code and reproduction resources**
>
> - Complete code: [KMnO4-zx/agentic-rl-lab/experiments/dapo](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/experiments/dapo)
> - Shared training entry point: [experiments/dapo/train.py](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/dapo/train.py)
> - Paper: [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
> - Official project page: [dapo-sia.github.io](https://dapo-sia.github.io/)
> - Official implementation: [BytedTsinghua-SIA/DAPO](https://github.com/BytedTsinghua-SIA/DAPO)
> - PyTRIO documentation: [https://docs.pytrio.com/docs](https://docs.pytrio.com/docs)

This is the sixth chapter in my series reproducing ten reinforcement learning algorithms.

Previous chapters cover:

- [Chapter 0: RL loss functions](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/loss-functions/readme.md)
- [Chapter 1: GRPO](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/grpo/readme.md)
- [Chapter 2: OPD](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/medical-opd/readme.md)
- [Chapter 3: Search-R1 for the price of a drink](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/search-r1/readme.md)
- [Chapter 4: OPSD for the price of a meal](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/opsd/readme.md)
- [Chapter 5: ReTool for the price of two coffees](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/retool/readme.md)

I originally planned the familiar route: explain the algorithm, finish training and organize the results.

Running it brought a different set of observations before any attractive reward curve:

> **Dynamic Sampling fills a training batch with more informative groups, but pays for that quality during rollout. When only one informative group is missing, repeated refill rounds become synchronization barriers.**

I want to document three things:

1. What DAPO changes relative to GRPO.
2. How to read and run the repository's DAPO and GRPO code.
3. Why Dynamic Sampling can be slow in practice and interfere with the time efficiency sought by asynchronous RL.

![](./images/dapo-paper.png)

## The findings first

The repository provides both DAPO and GRPO entry points so readers can inspect the algorithm switches and run them. This article presents no training-curve comparison and makes no claim that one algorithm is better.

My DAPO run had already taken about 4.17 hours by step 35. Finishing would cost substantially more time than expected, so I stopped the run and reduced the default configuration.

The important issue is visible in the code. Dynamic Sampling discards all-correct and all-incorrect groups and samples replacements. Discarded completions do not enter the PPO loss, but their generated tokens and waiting time have already been paid for. Refilling the final missing group can introduce repeated waiting barriers.

## What is DAPO?

**Decoupled Clip and Dynamic sAmpling Policy Optimization**, or DAPO, is a recipe of improvements for reinforcement learning with long chains of thought, built on the GRPO framework.

The official description highlights four changes:

1. **Clip-Higher:** loosen the upper PPO clipping bound on the positive-advantage side, allowing valuable low-probability tokens more room to increase.
2. **Dynamic Sampling:** filter prompts whose groups are all correct or all incorrect and keep sampling replacements.
3. **Token-level Policy Gradient Loss:** give completion tokens equal weight, rather than averaging each response first.
4. **Soft Overlong Punishment:** gradually penalize responses near the length limit, reducing reward noise from abrupt truncation.

This overview compares the four changes with GRPO:

![](./images/01 · DAPO vs GRPO Overview.png)

The implementation keeps two presets in the same [`train.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/dapo/train.py):

| Item | GRPO | DAPO |
| --- | --- | --- |
| PPO clip | `[0.8, 1.2]` | `[0.8, 1.28]` |
| Loss reduction | sample mean | token mean |
| Dynamic Sampling | Off | On |
| Soft Overlong Punishment | Off | On |

Both presets share the model, data, rollout and training framework, making their algorithmic differences easier to inspect.

## What problem does each change address?

### 1. Dynamic Sampling: train on groups with a relative signal

GRPO samples multiple answers to one question and computes within-group advantages:

```text
8 条全答对  → reward 方差为 0 → correctness 信号退化
8 条全答错  → reward 方差为 0 → correctness 信号退化
有对有错    → 可以比较哪些轨迹应该提高概率、哪些应该降低
```

DAPO removes groups that are entirely correct or entirely incorrect, then samples another question. [Section 3.2 of the paper](https://arxiv.org/pdf/2503.14476#page=5) continues sampling before training until the batch contains only informative groups with `accuracy ∈ (0, 1)`.

![](./images/02 · Bounded Dynamic Sampling.png)

The motivation is reasonable. If many prompts become too easy or too hard, fewer examples in a fixed batch contribute gradients. Dynamic Sampling performs online difficulty selection, concentrating training near the current policy's capability boundary.

This implementation adds an engineering limit:

```text
目标有效 group 数：B
最大候选 group 数：B × max_candidate_multiplier
当前默认 multiplier：2
```

Once the candidate budget is exhausted, it trains on the informative groups collected so far rather than refilling indefinitely. If none are collected, it skips the update. This `2×` cap and partial-batch behavior are implementation changes, distinct from original DAPO's strict full-batch semantics.

A key distinction: Dynamic Sampling filters by **raw correctness**, requiring both correct and incorrect answers within a group. Advantages use rewards after length penalties. Thus “zero-advantage group” here more precisely means a group with degenerate raw-correctness outcomes.

### 2. Clip-Higher: give positive-advantage tokens more room

PPO constrains the new-to-old policy probability ratio to limit updates. GRPO commonly uses symmetric bounds:

```text
clip_low  = 0.8
clip_high = 1.2
```

DAPO increases the upper bound to `1.28` while retaining the lower bound `0.8`. Intuitively, downward updates remain constrained while useful, low-probability tokens with positive advantage can increase more aggressively.

This targets entropy collapse. It does not increase every good response unconditionally: the clipped surrogate and advantage sign jointly determine each token's gradient.

### 3. Token Mean: longer answers contribute by token count

GRPO's sample mean is:

```text
每条回答先对 token loss 求均值 → 再对所有回答求均值
```

A 100-token response and a 4,000-token response therefore have equal outer weight in the batch.

DAPO's token mean is:

```text
把所有 completion token 合并 → 对全部 token 直接求均值
```

Every token has equal weight, so longer reasoning contributes more gradient terms. This avoids diluting long trajectories through sequence averaging, but also gives long responses more optimization weight. Interpret it alongside length shaping.

![](./images/03 · Clip-Higher and Token Mean.png)

### 4. Soft Overlong Punishment: penalize before abrupt truncation

A completion truncated at the token limit may never emit its final `\boxed{}` answer, abruptly changing a potentially correct result into an incorrect one. DAPO adds a linear penalty near the limit:

```text
penalty_start = max_tokens - overlong_cache

length <= penalty_start:  penalty = 0
penalty_start < length <= max_tokens:
                         penalty = -(length - penalty_start) / overlong_cache
length > max_tokens:      penalty = -1

shaped_reward = correctness_reward + length_penalty
```

![](./images/04 · Soft Overlong Punishment.png)

The figure uses the original configuration, `max_tokens=8192` and `overlong_cache=2048`. The penalty starts at 6,144 tokens and reaches −1 at 8,192.

The repository now defaults to `max_tokens=4096` and `overlong_cache=1024`, giving this penalty interval:

```text
0 ～ 3,072 tokens：不惩罚
3,584 tokens：     -0.5
4,096 tokens：     -1.0
```

The length penalty augments correctness reward. This implementation assigns +1 for correct and −1 for incorrect, then adds the penalty.

## How did we reproduce it?

### Run the code first

All code for this chapter is in `experiments/dapo/`.

The project requires Python `>=3.13`. PyTRIO performs sampling and LoRA training, so log in after installing dependencies.

Clone the project and enter the DAPO directory:

```bash
git clone https://github.com/KMnO4-zx/agentic-rl-lab.git
cd agentic-rl-lab

uv sync
trio login
swanlab login
cd experiments/dapo
```

Download and prepare training data:

```bash
uv run python prepare_data.py
```

The script writes `datasets/train.jsonl` and `datasets/dev.jsonl` beneath the current directory.

Launch the reduced default DAPO configuration:

```bash
uv run python train.py \
    --algorithm dapo \
    --max-steps 10 \
    --groups-per-step 4 \
    --group-size 8 \
    --max-candidate-multiplier 2 \
    --max-prompt-tokens 512 \
    --max-tokens 4096 \
    --overlong-cache 1024 \
    --swanlab-mode disabled
```

To inspect the preset differences with the same framework, change the algorithm to `grpo`:

```bash
uv run python train.py \
    --algorithm grpo \
    --max-steps 10 \
    --groups-per-step 4 \
    --group-size 8 \
    --max-prompt-tokens 512 \
    --max-tokens 4096 \
    --overlong-cache 1024 \
    --swanlab-mode disabled
```

With the code running, we can examine the implementation.

### Code structure

Four files contain the main implementation:

- [`prepare_data.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/dapo/prepare_data.py): download, clean and pin the dataset.
- [`rollout.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/dapo/rollout.py): group rollouts, advantages and Dynamic Sampling.
- [`reward.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/dapo/reward.py): mathematical answer checking and Soft Overlong Punishment.
- [`train.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/dapo/train.py): shared GRPO/DAPO PPO updates.

### Data

Training uses the official [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) dataset.

The official Parquet files contain many duplicate rows. Preparation therefore:

1. Removes the original `Answer:` wrapper to avoid conflicting with the required `\boxed{}` format.
2. Normalizes questions and deduplicates them.
3. Removes entire groups when the same question has conflicting ground truths.
4. Selects 50 development examples with a fixed seed and uses the remainder for training.

The resulting counts are:

```text
train.jsonl：17,126 道题
dev.jsonl：      50 道题
```

The dataset revision is pinned in the script so upstream changes cannot silently alter the training set for the same command.

### Rewards and advantages

The last `\boxed{}` within a completion's final 300 characters is extracted, then checked for mathematical equivalence with `math_verify`:

```text
正确：+1
错误或格式非法：-1
DAPO：再叠加 Soft Overlong Punishment
```

The 8 completions for each question use shaped rewards for within-group standardization:

```text
A_i = (r_i - mean(r)) / (std(r) + epsilon)
```

Prompt tokens are masked; only model-generated completion tokens enter the PPO loss.

### The initial DAPO configuration

My initial run used:

| Item | Configuration |
| --- | --- |
| Base Model | `Qwen/Qwen3.5-4B` |
| Training method | LoRA rank 32 |
| Target prompt groups per step | 4 |
| Completions per question | 8 |
| Target completions per step | 32 |
| Prompt limit | 1,024 tokens |
| Completion limit | 8,192 tokens |
| Soft Overlong interval | Final 2,048 tokens |
| Maximum DAPO candidates | 2× the target batch |
| rollout concurrency | 16 |
| optimizer | Adam，lr `4e-5`，β `(0.9, 0.95)` |
| seed | 42 |

This configuration ran the algorithm but exposed large timing and per-step token variation. The defaults were reduced to make learning and trial runs lighter:

```text
max_prompt_tokens：1,024 → 512
max_tokens：       8,192 → 4,096
overlong_cache：   2,048 → 1,024
groups_per_step：  4（保持不变）
group_size：       8（保持不变）
candidate cap：    2×（保持不变）
```

This is smaller than the paper's experiments and is intended to inspect the mechanisms and implementation issues.

## The practical problem: waiting for the last informative group

“Filter uninformative examples and refill the batch” sounds simple. In practice, refilling the final part can be the slowest step.

Suppose each update requires `B=4` informative groups:

```text
第一轮并发采 4 组
├── 第 1 组：有效
├── 第 2 组：有效
├── 第 3 组：有效
└── 第 4 组：全对或全错，被丢弃
```

Only 1 group remains missing. The current implementation samples 1 replacement group instead of another full set of 4:

```text
补采第 5 组 → 无效 → 再等
补采第 6 组 → 无效 → 再等
补采第 7 组 → 无效 → 再等
补采第 8 组 → 仍可能无效，候选预算耗尽
```

Each wave uses `asyncio.gather` and a concurrency semaphore, but the next wave's size is unknown until all previous completions finish and rewards are calculated. Near a full batch, Dynamic Sampling can form a chain of serial refill barriers.

For target batch size `B` and candidate multiplier `M`:

```text
最多采样的 group 数 = M × B
候选 token 计算量上限约为正常 batch 的 M 倍

如果第一轮只差 1 个有效 group：
最多经历的 rollout wave 数 = 1 + (M - 1) × B
```

With the current `B=4, M=2`, the worst case is more than two waves:

```text
1 轮初始 batch + 4 轮单 group 补采 = 5 个等待 wave
```

At `3×`, the worst case can reach `1 + 2×4 = 9` waves. Each wave can include a long completion near 8,192 tokens, so wall-clock amplification is not necessarily bounded by `2×/3×`.

This is the structural tension I see between Dynamic Sampling and asynchronous RL:

- Asynchronous RL aims to keep rollouts flowing and move completed data to the next stage promptly.
- Strict Dynamic Sampling requires the current step to collect `B` informative groups first.
- The learner waits for the final replacement group before updating.
- Concurrency narrows as the batch fills, making the last informative group a potential straggler for the whole update.

> [DAPO paper, Section 3.2](https://arxiv.org/pdf/2503.14476#page=5): “Before training, we keep sampling until the batch is fully filled with samples whose accuracy is neither 0 nor 1.”

Original DAPO fills the informative batch before updating, even if only one group is missing. The paper also notes that generation in synchronous systems without generation pipelining is often dominated by long-tail samples, so Dynamic Sampling need not reduce training efficiency.

In this implementation's synchronous loop, however, each group must finish and receive rewards before another refill can be scheduled. Later refill waves cannot overlap the preceding wave's straggler generation. Every extra refill therefore adds another wait, particularly when only one group is missing. Here, long-tail waiting did not hide the extra sampling cost; refills substantially increased training time.

## Evaluate Dynamic Sampling by both quality and budget

A filtered completion still incurs work:

```text
prefill → autoregressive sampling → reward parsing → correctness verification
```

It simply does not enter the final PPO loss.

Track at least four dimensions when running this code:

| Dimension | Recorded metric | Question answered |
| --- | --- | --- |
| Data quality | `rollout/effective_fill_ratio` | How much of the target batch was filled? |
| Selection cost | `rollout/oversample_ratio` | How many extra questions produced those informative groups? |
| Token utilization | `train_completion_tokens / completion_tokens` | What fraction of generated tokens entered training? |
| Time efficiency | `time/step_seconds` | How long did one parameter update take? |

A fuller effective batch must be considered alongside extra candidates, discarded rollout tokens and actual update latency.

## Another practical issue: cost peaks from long sequences and dynamic batches

Training time is difficult to estimate in advance:

- The completion limit was 8,192, with limit-reaching answers in every step.
- Candidate groups, informative groups and training-token counts vary with refill outcomes.
- Long-tail answers can prolong refill rounds when the final group is missing.

Reducing the default completion limit to 4,096 also reduces per-step token and wall-clock variation. A future long-context run should bound **step time**, **total candidate tokens** and **training tokens**, in addition to group count.

Both DAPO and GRPO code are available for further study. This article examines Dynamic Sampling's behavior and engineering costs; the unfinished DAPO run is not an effectiveness comparison.

## What should you watch in your own run?

Track training benefit alongside sampling cost, rather than reward alone:

```text
time/step_seconds

rollout/candidate_groups
rollout/effective_groups
rollout/effective_group_ratio
rollout/effective_fill_ratio
rollout/oversample_ratio

rollout/completion_tokens
rollout/train_completion_tokens
rollout/max_completion_tokens

train/update_skipped

reward/accuracy
reward/length_penalty_mean
reward/shaped_mean

ppo/clip_fraction
ppo/lower_clip_fraction
ppo/upper_clip_fraction
ppo/gradient_active_tokens
```

Pay particular attention to these three relationships:

```text
effective_groups / candidate_groups  → 数据筛选效率
train tokens / rollout tokens        → token 利用率
effective_groups / step_seconds      → 真正的 wall-clock 产出
```

`effective_fill_ratio` alone cannot establish whether Dynamic Sampling is worthwhile. Include the timing measurements to see what those informative groups cost.

## Summary

DAPO's four changes address concrete long-CoT RL issues: entropy collapse, uninformative groups, long-trajectory weighting and truncation-related reward noise.

Dynamic Sampling is especially appealing, and also exposed the clearest engineering cost here:

```text
它让每次更新看到更多有效 group，
但为了等到最后一个有效 group，
rollout 会反复补采、反复等待，
最终把动态筛选的成本完整付在 wall-clock 上。
```

## Reflections

> ***Every gift of fate already carries a hidden price.***

Implementing DAPO from scratch changed how I diagnose slow training. With transformers or verl, I would usually suspect the hardware or configuration because much of the trainer felt like a black box. Reviewing this DAPO implementation line by line let me see what it was doing and why it waited. For me, that is PyTRIO's appeal: I can implement and inspect the core algorithm without also building the combined training/inference infrastructure. The main question becomes whether the algorithm itself is implemented correctly.

## References

### Paper and official implementation

1. Qiying Yu et al. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476), 2025.
2. [DAPO project page](https://dapo-sia.github.io/)
3. [BytedTsinghua-SIA/DAPO implementation](https://github.com/BytedTsinghua-SIA/DAPO)

### Data and model

1. [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
2. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

### This implementation and experiment

1. [Complete DAPO PyTRIO code](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/experiments/dapo)
2. [PyTRIO documentation](https://docs.pytrio.com/docs)
