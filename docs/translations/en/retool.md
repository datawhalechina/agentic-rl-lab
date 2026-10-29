# Reproducing Ten RL Algorithms, Chapter 5: Trying ReTool for the Price of Two Coffees

> Ready to run the code? See the [quick start](./start.md): prepare data → train → evaluate.

![ReTool experiment overview](./images/封面.png)

<div align="center">
  <a href="https://www.zhihu.com/people/feng-qi-xia-pian" target="_blank"><img alt="Zhihu" src="https://img.shields.io/badge/Zhihu-知乎-4362f6"></a>
  <a href="https://www.xiaohongshu.com/user/profile/63c2055e000000002502c58c" target="_blank"><img alt="Rednote" src="https://img.shields.io/badge/Rednote-小红书-e93c49"></a>
  <a href="https://github.com/KMnO4-zx/agentic-rl-lab"><img alt="visitors" src="https://komarev.com/ghpvc/?username=KMnO4-zx-agentic-rl-lab-retool&amp;label=visitors&amp;color=1283c3&amp;style=flat"></a>
</div>

> **Code and reproduction resources**
>
> - Complete code: [KMnO4-zx/agentic-rl-lab/experiments/retool](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/experiments/retool)
> - Training script: [experiments/retool/train.py](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/retool/train.py)
> - SwanLab: [Complete training records](https://swanlab.cn/@kmno4/llm-agent-rl-lab-retool/overview)
> - Paper: [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)
> - PyTRIO documentation: [https://docs.pytrio.com/docs](https://docs.pytrio.com/docs)
> - What is PyTRIO? [Introduction on Zhihu](https://zhuanlan.zhihu.com/p/2063265307226019219)
> - PyTRIO Skill：[SwanHubX/pytrio-skill](https://github.com/SwanHubX/pytrio-skill)

This is the fifth chapter in my series reproducing ten reinforcement learning algorithms.

Earlier chapters cover:

- [Chapter 0: RL loss functions](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/loss-functions/readme.md)
- [Chapter 1: GRPO](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/grpo/readme.md)
- [Chapter 2: OPD](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/medical-opd/readme.md)
- [Chapter 3: Search-R1 for the price of a drink](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/search-r1/readme.md)
- [Chapter 4: OPSD for the price of a meal](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/opsd/readme.md)

This time the tool does computation: **ReTool lets the model decide when to write Python, and what to calculate, within a long mathematical reasoning process.**

In plain language:

> The model works through a problem, runs code when calculation becomes difficult, and is graded only on its final answer. It learns when a calculator helps through outcomes.

The initial pipeline check cost about as much as two coffees.

I first ran a 20-step trial to verify the loop. Its bill was:

```text
训练会话（20 step）:  ¥22.89   （4.34M prefilling / 1.66M train / 1.77M sample）
AIME25 评测 × 2:     ¥4.16 + ¥4.61   （Base、Step 20）
合计:                ¥31.66
```

![](./images/pytrio-consume-20step.png)

At the time, a Luckin coffee cost roughly `13–16` yuan, or `26–32` for two. That comparison describes the small trial used to decide whether to continue.

Behavior already changed during those 20 steps:

```text
correct（前 10 步均值 → 后 10 步均值）:  0.289 → 0.389
code_calls:                              1.24 → 1.72
degenerate group（整组无梯度）:           0
```

Evaluation also improved over the same interval. The later results section expands this figure; first compare Base with Step 20:

![](./images/checkpoint_avg_pass_format.png)

```text
Base Model:  Average@12 23.61% · Format 25.83%
Step 20:     Average@12 32.22% · Format 36.67%
```

After that check, I used the same configuration for a complete 200-step training run:

```text
训练会话（200 step）:  ¥224.44   （48.67M prefilling / 15.97M train / 15.84M sample）
```

![](./images/pytrio-comsume-200step.png)

Evaluation is separate: Base plus 5 checkpoints gives 6 AIME25 evaluations, each with 30 questions and 12 samples per question, costing about `¥4–5` each in this experiment.

The attributable costs together are:

```text
训练（验证跑 + 200 步轨迹）:  ¥22.89 + ¥224.44 = ¥247.33
评测（截图中 4 次）:           ¥18.49   （另有 2 次未入镜，每次约 ¥4～5）
```

## Where does ReTool come from?

ReTool is work released by ByteDance's Seed team in April 2025, titled:

> ReTool: Reinforcement Learning for Strategic Tool Use in LLMs

It addresses a direct problem.

Long-chain reasoning improves mathematical performance, but purely textual reasoning must perform complex calculations internally. Roots, high-degree equations or large enumerations can introduce unnoticed arithmetic errors during a long solution.

ReTool allows the model to write code during reasoning. Two central settings are:

1. **Code-interlaced rollout:** pause generation at code, execute it in a sandbox, append stdout/stderr to context and continue. Interpreter-output tokens do not enter the loss.
2. **Outcome-only reward:** reward a correct final answer, without rewarding code use itself. When and how often to use code is learned through results.

The paper also includes cold-start SFT on code-interlaced demonstrations, an asynchronous SandboxFusion sandbox and KV-cache reuse. This reproduction distinguishes what it retains from what it omits.

## ReTool in everyday terms

Imagine a student taking a math exam with a calculator that clears after every use:

- The student writes the reasoning and can run code for difficult calculations.
- The calculator has no persistent state; results must be printed explicitly.
- Grading checks the final boxed answer: +1 for correct and −1 for incorrect.

No one specifies exactly when to use the calculator. The student may initially overuse or underuse it. Outcome rewards can favor a policy that handles reliable calculations directly and delegates harder ones to code.

## How did we reproduce it?

The implementation retains the central algorithm around three components:

- **Tool protocol** ([`protocol.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/retool/protocol.py)): declare the interpreter as a native model tool, define the system prompt and parse `<tool_call>`.
- **Local execution sandbox** ([`sandbox.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/retool/sandbox.py)): execute each snippet in a fresh Python subprocess with timeouts and resource limits.
- **Outcome reward** ([`reward.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/retool/reward.py)): check mathematical equivalence of the final `\boxed{}` answer, scoring +1 / −1.

The multi-turn rollout state machine, group advantages and PyTRIO loop reuse the framework verified in earlier chapters.

The architecture diagram shows the budgets and hard limits for each stage:

![](./images/ReTool.png)

Key decisions are:

- **Use the native tool protocol.** Code is sent through Qwen's `<tool_call>` format, and stdout/stderr returns in a `role: "tool"` message. This provides code-interlaced interaction without adding tokens or changing the tokenizer and chat template.
- **Skip cold-start SFT and train Base directly with RL.** This is the largest omitted stage. In the base-model smoke test, 2 questions × 8 trajectories yielded an `87.5%` tool-use rate (14/16), `85.4%` valid-call rate and mixed rewards within groups, enough to test RL without teaching the format first.
- **Match strict answer checking:** inspect the final 300 characters, extract the last `\boxed{}`, and use `math_verify` for equivalence. Correct scores +1; incorrect or malformed scores −1.
- **Use PyTRIO's built-in `ppo`**, with bounds `0.8 / 1.28`, corresponding to the recipe's ε_low=0.2 / ε_high=0.28, or DAPO Clip-Higher.

## Experimental results

Evaluation fixes AIME25's 30 questions and 12 samples per question. Base and every checkpoint share the same configuration:

```text
temperature = 1.0
top_p = 0.7
val_n = 12（每题 12 次采样）
轨迹预算 = 8,192 tokens（与训练一致）
模式 = retool（启用 code 工具）
```

Results across one 200-step training run are:

![](./images/checkpoint_avg_pass_format.png)

| Model / checkpoint | Average@12 | Pass@12 | Format | Mean code calls | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B Base | 23.61% | 50.00% | 25.83% | 1.69 | 2.69 |
| Step 20 | 32.22% | 63.33% | 36.67% | 1.37 | 2.37 |
| Step 50 | 41.11% | 63.33% | 52.78% | 1.70 | 2.70 |
| Step 100 | 44.72% | 66.67% | 57.78% | 1.66 | 2.66 |
| Step 150 | 46.11% | 73.33% | 70.83% | 2.43 | 3.43 |
| **Step 200** | **47.50%** | **63.33%** | **76.11%** | 1.98 | 2.98 |

The metrics mean:

- **Average@12:** the proportion correct among 360 generations, estimating single-sample accuracy.
- **Pass@12:** the proportion of 30 questions answered correctly at least once across 12 samples.
- **Format:** the proportion of generations with a valid `\boxed{}` answer.

Step 200 versus Base:

```text
Average@12:  23.61% → 47.50%  （+23.89 个百分点）
Pass@12:     50.00% → 63.33%  （+13.33 个百分点）
Format:      25.83% → 76.11%  （+50.28 个百分点）
```

Correct generations increase from `85` to `171`. Questions answered correctly at least once increase from `15` to `19`, with a peak of `22` at Step 150.

> Across this run, Average@12 increases at every evaluated checkpoint and Format doubles. However, 30 questions, one training run and no multi-seed study cannot establish a stable reproduction of the paper's gains. Pass@12 falls from 73.33% at Step 150 to 63.33% at Step 200, illustrating the variability of this small evaluation.

The 8,192-token trajectory budget truncates some long AIME reasoning and limits format completion. The paper uses a 16k budget.

## Reproduction configuration

The trial and 200-step run share the same configuration apart from `--max-steps`:

| Item | This implementation |
| --- | --- |
| Base Model | `Qwen/Qwen3.5-4B` |
| Training | LoRA rank 32 through the PyTRIO training client |
| Training data | `BytedTsinghua-SIA/DAPO-Math-17k` → `datasets/train.jsonl` |
| Per step | 8 questions × group size 8 = 64 trajectories |
| Turn budget | `max_code_calls=4`, `max_assistant_turns=6` |
| Token budget | Trajectory ≤ 8,192; assistant turn ≤ 1,024; tool output ≤ 512 |
| Sampling | Temperature 1.0, top_p 1.0 |
| Sandbox | Local subprocess, 30s timeout, 8 workers |
| Optimizer | Adam, lr `4e-5`, β (0.9, 0.95) |
| Loss | `ppo`，clip 0.8 / 1.28 |
| Advantage | Within-group centering; observation tokens receive 0 |
| Checkpoints | Every 50 steps in the full run; every 5 in the trial |
| Seed | 42 |

The official recipe uses `train_batch_size=512`, `n_resp_per_prompt=16`, max sequence length 16384, `max_turns=8` and actor lr `1e-6`; the paper fully trains Qwen2.5-32B. We use smaller budgets and LoRA. The released code also reduces incorrect-answer penalties according to tool turns (`score = min(0, score + (num_turns - 2) / 2 * 0.1)`). This differs from the paper's outcome-only description; this reproduction follows the paper and omits that shaping.

## The ReTool training loop in detail

One training step can be divided into seven parts.

### Step 1: prepare math questions with verifiable answers

[`prepare_data.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/retool/prepare_data.py) follows the recipe's [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k), exporting `question + answer` JSONL:

```json
{"id": "40f50547-...", "question": "The points $P,$ $Q,$ and $R$ are represented by ...", "answer": "3", "data_source": "math_dapo"}
```

The original DAPO prompt includes an `"Answer:"` wrapper. We remove it to avoid conflicting with the `\boxed{}` protocol.

### Step 2: declare the interpreter as a native tool

[`protocol.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/retool/protocol.py) declares one tool through the chat template's `tools` argument:

```python
"name": "code_interpreter",
"parameters": {"code": "The Python code to be executed."}
```

The system prompt specifies the task, available tool and invocation format. It also states code constraints: finish within seconds, use little memory, avoid file I/O, bound enumeration, print results and assume stateless execution. The template fixes `enable_thinking=False`. The original paper's base model has no thinking-mode switch; its long reasoning is ordinary generated text. The official reproduction also reports poor code use and training behavior with Qwen3 thinking mode.

### Step 3: branch and continue trajectories

In [`rollout.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/retool/rollout.py), the first request uses `num_samples=group_size` to create 8 trajectories. Later requests continue each branch independently. Generation pauses at `<tool_call>`; after execution, [`build_next_prompt`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/retool/protocol.py#L127) appends assistant closing tokens and the tool observation to the actual sampled tokens, then generation resumes.

Continuation keeps the original token sequence without text↔token re-encoding. The official implementation reports that irreversible re-encoding can cause collapse around 100 steps and NaN gradient norms; this design avoids that mismatch.

### Step 4: execute code locally

[`sandbox.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/retool/sandbox.py) replaces SandboxFusion with local subprocess execution, avoiding a separate cloud-sandbox service:

```text
python -B -c <code>   # 每次调用起全新进程
list 形式 argv        # 无 shell、无脚本文件、天然无状态
30s wall-clock 超时    # os.killpg 杀死整个进程组
RLIMIT_CPU            # 内核级 CPU 时间兜底
stdout/stderr 截断     # 单次回包最多 512 tokens
```

Two details follow the official approach. The final nonempty line receives automatic `print()` handling, and tool output is sanitized before entering context. Printed special tokens such as `<|im_end|>` would otherwise corrupt observation structure.

### Step 5: reward only the final answer

[`reward.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/retool/reward.py) uses strict final-answer checking consistent with `compute_score(strict_box_verify=True)`:

```text
取回答最后 300 个字符 → 提取最后一个 \boxed{} → math_verify 数学等价
对 +1 / 错 −1（找不到合法 \boxed{} 也算错）
```

Code use itself receives no reward. The tool policy develops through outcome rewards.

### Step 6: retain observations as context and mask their loss

Sandbox stdout/stderr remains visible to the model but is not model-generated. When [`train.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/retool/train.py) builds a Datum, observation positions receive zero advantage:

```python
advantages_by_token.extend([0.0] * len(delta_observation))
advantages_by_token.extend(
    [trajectory.advantage] * len(turn.completion_tokens)
)
```

Only assistant-generated tokens are optimized.

### Step 7: center group advantages and make one PPO update

For the 8 trajectories of a question, [`assign_group_advantages`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/retool/rollout.py#L273) centers rewards within the group and separately records degenerate groups. In the 8-question batches of this run, this did not eliminate all gradients. The full batch then receives one update:

```python
loss_fn="ppo",
loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}
```

Micro-batches use dynamic packing. Before submission, advantages are scaled by sample proportions so accumulated gradients match a global sample mean.

Two implementation pitfalls appeared during development:

1. **Qwen3.5's template strips assistant content** and splits text containing `</think>` into reasoning/content fields. Locating the assistant boundary by rerendering actual output therefore fails. `build_next_prompt` uses placeholder content for canonical boundary calculations; the added segment depends only on the tool message.
2. **Unsanitized tool output can corrupt observation structure**, as described in Step 4. This was caught during review.

## SwanLab training records

Rewards, correctness, code calls, trajectory lengths and sandbox diagnostics are recorded in [SwanLab](https://swanlab.cn/@kmno4/llm-agent-rl-lab-retool/overview).

Reward curves across 200 steps:

![](./images/swanlab-reward.png)

```text
reward/mean:    -0.5 → +0.3
reward/correct:  0.2 → 0.6
reward/format:  0.25 → 0.85
```

Rollout behavior:

![](./images/swanlab-rollout.png)

```text
rollout/valid_tool_call_rate:  0.6 → 0.95    # 模型越调越规范
rollout/code_calls:            1.0 → 2.0     # step 90 附近峰值约 2.6
rollout/turns:                 2.0 → 2.8
rollout/trajectory_tokens:     稳定在 ~2,200 # 训练分布上 8k 预算宽裕
rollout/degenerate_group_rate: 缓降           # 无零梯度灾难
```

Sandbox diagnostics:

![](./images/swanlab-sandbox.png)

```text
sandbox/success_rate:  0.68 → 0.80
sandbox/error_rate:    0.32 → 0.18   # 模型写的代码越来越能跑
sandbox/timeout_rate:  稳定在 2～3%
sandbox/latency:       0.36s → 1.4s  # 代码变复杂，执行时间自然变长
```

The curves show increasingly valid calls (0.6 → 0.95), a roughly halved execution-error rate and improving correctness (0.2 → 0.6). Together they are consistent with better tool invocation, executable code and useful calculations. Truncation under the 8k budget remains a limitation for long reasoning and final-answer formatting.

## Running this reproduction

### 1. Install and log in

```bash
git clone https://github.com/KMnO4-zx/agentic-rl-lab.git
cd agentic-rl-lab

uv sync
trio login
swanlab login
```

The local machine needs a CPU environment. PyTRIO handles remote sampling and LoRA training; the interpreter runs in local subprocesses.

### 2. Download and prepare training data

```bash
uv run python experiments/retool/prepare_data.py
```

The script downloads `BytedTsinghua-SIA/DAPO-Math-17k` and exports:

```text
experiments/retool/datasets/
├── raw/dapo-math-17k.parquet   # 原始数据
├── train.jsonl                  # 训练集（question + answer）
└── dev.jsonl                    # 50 条开发集
```

### 3. Run the 20-step trial

```bash
uv run python experiments/retool/train.py \
    --max-steps 20 \
    --save-every 5 \
    --run-name retool-qwen35-4b-step20
```

The documented 20-step run showed increasing correctness and code calls, with no degenerate groups, providing an initial pipeline check.

### 4. Run 200 training steps

```bash
uv run python experiments/retool/train.py \
    --max-steps 200 \
    --save-every 50 \
    --run-name retool-qwen35-4b-step200
```

Every 50 steps, save a resumable state and sampler weights for sampling/evaluation.

### 5. Evaluate Base and checkpoints

```bash
# Base Model
uv run python experiments/retool/eval.py \
    --mode retool \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --output experiments/retool/eval-results/aime25-retool-base.jsonl

# checkpoint（填入训练日志里的 trio:// 路径）
uv run python experiments/retool/eval.py \
    --mode retool \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --model-path trio://<your_sampler_weights_path> \
    --output experiments/retool/eval-results/aime25-retool-step200.jsonl
```

Evaluation also supports `--mode text`, disabling tools for a text-only baseline.

### 6. Plot checkpoint comparisons

Once `eval-results/` contains Base and checkpoint JSONL files:

```bash
uv run python experiments/retool/analysis.py
```

The figure is written to `experiments/retool/images/checkpoint_avg_pass_format.png`.

## Reproduction boundaries

- **8k trajectory budget versus 16k in the paper:** a major budget reduction. Long AIME reasoning can be truncated, and Format reaches roughly 76% in this run. A larger budget is a useful next experiment.
- **Local subprocess limits are not a cloud-sandbox boundary:** `python -c`, a 30s timeout and `RLIMIT_CPU` handle many accidental loops and timeouts. File I/O restrictions rely on the prompt, and memory limits on macOS are unreliable. This implementation is not intended for unattended execution of untrusted code.
- **Cold-start SFT is skipped:** Qwen3.5-4B's initial tool-use rate made that workable here; another base model may require it.
- **No dual-clip:** PyTRIO's built-in `ppo` does not expose the recipe's `clip_ratio_c=10.0`, which is omitted.
- **Evaluation scope:** 30 questions × 12 samples, one training run and one seed. Pass@12 can vary by one or two questions at this scale.

## Summary

ReTool reused much of the earlier multi-turn infrastructure, leaving three main new components: protocol, sandbox and reward. The resulting capability is concrete: choosing when to stop calculating internally and use code.

Three observations summarize the experiment:

```text
两杯瑞幸（¥31.66）的 20 步验证跑，就足够确认 ReTool 闭环有效；
一条 200 步轨迹（¥224.44）下来，AIME25 Average@12 从 23.61% 提到 47.50%；
模型学会的顺序是：先调规范，再写能跑的代码，最后写对解题有用的代码。
```

For me, the small trial showed how native model tools plus a local execution environment make Agentic RL ideas practical to investigate from a MacBook Air. The two-coffee comparison applies to that initial trial, with full training and evaluation accounted for separately above.

## References

### Paper and official implementation

1. Jiazhan Feng et al. [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536), 2025.
2. [verl `recipe/retool`, official reproduction](https://github.com/volcengine/verl/tree/main/recipe/retool)
3. [Volcengine veRL reproduction guide](https://developer.volcengine.com/articles/7545026392128225323)
4. [swordfaith/ReTool-SFT-multi-turn, cold-start data](https://huggingface.co/datasets/swordfaith/ReTool-SFT-multi-turn)

### Data and model

1. [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
2. [yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025)
3. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

### This implementation

1. [Complete ReTool PyTRIO code](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/experiments/retool)
2. [PyTRIO documentation](https://docs.pytrio.com/docs)
3. [What is PyTRIO? Introduction on Zhihu](https://zhuanlan.zhihu.com/p/2063265307226019219)
4. [SwanLab record for this experiment](https://swanlab.cn/@kmno4/llm-agent-rl-lab-retool/overview)
5. [SwanLab documentation](https://docs.swanlab.cn/)
