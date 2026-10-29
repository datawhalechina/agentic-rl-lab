# Reproducing Ten RL Algorithms, Chapter 4: OPSD Training for the Price of a Meal

> Ready to run the code? See the [quick start](./start.md): prepare data → train → evaluate.

![](./images/head.png)

<div align="center">
  <a href="https://www.zhihu.com/people/feng-qi-xia-pian" target="_blank"><img alt="Zhihu" src="https://img.shields.io/badge/Zhihu-知乎-4362f6"></a>
  <a href="https://www.xiaohongshu.com/user/profile/63c2055e000000002502c58c" target="_blank"><img alt="Rednote" src="https://img.shields.io/badge/Rednote-小红书-e93c49"></a>
  <a href="https://github.com/datawhalechina/agentic-rl-lab"><img alt="visitors" src="https://komarev.com/ghpvc/?username=KMnO4-zx-agentic-rl-lab-opsd&amp;label=visitors&amp;color=1283c3&amp;style=flat"></a>
</div>

> **Code and reproduction resources**
>
> - Complete code: [datawhalechina/agentic-rl-lab/experiments/opsd](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/opsd)
> - Asynchronous training: [01-opsd-async.py](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/opsd/01-opsd-async.py)
> - SwanLab: [Complete training curves](https://swanlab.cn/@kmno4/llm-agent-rl-lab-opsd/runs/pxsnuza4/chart)
> - Paper: [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734)
> - Official implementation: [siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD)
> - PyTRIO documentation: [https://docs.pytrio.com/docs](https://docs.pytrio.com/docs)
> - What is PyTRIO? [Introduction on Zhihu](https://zhuanlan.zhihu.com/p/2063265307226019219)
> - PyTRIO Skill：[SwanHubX/pytrio-skill](https://github.com/SwanHubX/pytrio-skill)

This is the fourth chapter in my series reproducing ten reinforcement learning algorithms.

Previous chapters cover:

- [Chapter 0: RL loss functions](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/loss-functions/readme.md)
- [Chapter 1: GRPO](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/grpo/readme.md)
- [Chapter 2: OPD](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/medical-opd/readme.md)
- [Chapter 3: Search-R1 for the price of a drink](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/search-r1/readme.md)

This time I wanted to try **OPSD, On-Policy Self-Distillation**, a then-recent method with an appealing idea.

In plain language:

> Let the same model answer closed-book, then evaluate its own tokens with the reference solution available, without requiring a larger teacher.

Having already run GRPO, OPD and Search-R1 with PyTRIO, I wanted to focus this reproduction on the dual prompts, teacher log probabilities and sampled-token loss. Reading the paper, implementing it, training for 100 steps, evaluating Base and 4 checkpoints, and plotting the results took less than two days.

The full training session cost `41.81` yuan on PyTRIO:

![](./images/pytrio-consume.png)

That `41.81` yuan covers only the 100-step training run, excluding the full AIME25 evaluations.

Evaluation cost more. We evaluated Base, Step 25, Step 50, Step 75 and Step 100: 5 model states. Each evaluation contains:

```text
30 道题 × 每题 12 次采样 = 360 条 completions
每条 completion 的 max_tokens = 38,912
```

The maximum generation budget per model state is therefore:

```text
30 × 12 × 38,912 = 14,008,320 tokens
```

`14.01M` is an upper bound, not actual usage. Models usually emit EOS before `max_tokens`; actual sample usage was `3.40M–3.86M tokens` per evaluation:

![](./images/sample-consume.png)

| Evaluated model | Actual sample tokens | Cost |
| --- | ---: | ---: |
| Base Model | 3.49M | ¥16.15 |
| Step 25 | 3.40M | ¥15.79 |
| Step 50 | 3.53M | ¥16.37 |
| Step 75 | 3.77M | ¥17.44 |
| Step 100 | 3.86M | ¥17.86 |
| **Total for 5 evaluations** | **18.05M** | **¥83.61** |

Training plus these 5 full evaluations gives an attributable cost of `41.81 + 83.61 = 125.42` yuan.

The title's meal-price comparison refers to running OPSD training. Obtaining the complete Pass@12 curve for Base plus 4 checkpoints costs more.

## Reproduction results

We use the same 30 questions with 12 samples per question. Base and every checkpoint share this configuration:

```text
temperature = 1.0
top_p = 1.0
top_k = -1
max_tokens = 38,912
thinking = off
```

Results:

![](./images/aime25-opsd-progress.png)

| Model / checkpoint | Average@12 | Pass@12 | Correct generations | Questions with at least one correct answer | Format rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B Base | 51.67% | 80.00% | 186 / 360 | 24 / 30 | 100% |
| Step 25 | 51.11% | 73.33% | 184 / 360 | 22 / 30 | 100% |
| Step 50 | 50.28% | 86.67% | 181 / 360 | 26 / 30 | 100% |
| Step 75 | 48.61% | 76.67% | 175 / 360 | 23 / 30 | 100% |
| **Step 100** | **52.78%** | **86.67%** | **190 / 360** | **26 / 30** | **100%** |

The two main metrics mean:

- **Average@12:** the fraction correct among 360 generations.
- **Pass@12:** the fraction of 30 questions answered correctly at least once across 12 samples.

Step 100 has the highest Average@12 among these checkpoints:

```text
Average@12: 51.67% → 52.78%  （+1.11 个百分点）
Pass@12:    80.00% → 86.67%  （+6.67 个百分点）
```

That corresponds to correct generations increasing from `186` to `190`, and questions answered correctly at least once increasing from `24` to `26`.

The gain is positive but small. Intermediate checkpoints do not improve monotonically; Step 75 Average@12 is below Base. These results support a limited conclusion:

> The complete OPSD training pipeline runs, and the final checkpoint improves slightly on this evaluation. Thirty questions, one training run and no multi-seed study are insufficient to claim a stable reproduction of the paper's gains.

## What is OPSD?

OPSD comes from a paper by Siyan Zhao and colleagues:

> Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

It addresses two practical problems.

The first concerns SFT.

During SFT, the model sees expert-written trajectories:

```text
problem → expert solution
```

At inference, however, it sees its own generated prefixes. An early error can shift the later context into states absent from expert demonstrations. This is a training–inference distribution mismatch.

The second concerns RLVR methods such as GRPO.

GRPO learns from the model's own rollouts but commonly receives an outcome reward only when a complete response ends:

```text
答对 → 1
答错 → 0
```

A reasoning trajectory thousands of tokens long then receives one sparse signal. If all 8 responses to a question are correct or all are incorrect, group-relative advantages also become zero.

OPSD combines these ideas:

- The current student generates its own rollout, maintaining on-policy data.
- The teacher supervises each token along that actual student trajectory.
- Teacher and student begin from the same initial model; a larger teacher is unnecessary.
- The teacher sees the reference solution, while the student does not.

The paper's full process is:

![](./images/opsd.png)

Think of the same student performing two roles.

First, answer closed-book:

```text
Student：只看题目，自己尝试解题。
```

Then, review with the solution available:

```text
Teacher：看到题目、参考解答，以及 Student 已经写出的前缀，
         判断 Student 的下一个 token 是否合理。
```

A crucial distinction is:

> The teacher does not generate a replacement answer. Only the student generates the trajectory; the teacher performs a forward pass on those actual tokens to obtain their log probabilities.

Training therefore uses prefixes visited by the current student, rather than imitation of a fixed collection of expert answers.

## How can the same model be both student and teacher?

The key is the prompt.

Both use the Qwen3.5-4B tokenizer and start from the same initial model, but receive different context:

![](./images/prompt.png)

The student sees only the question and output instructions:

```text
Problem: {problem}

Please reason step by step, and put your final answer within \boxed{}.
```

The teacher additionally sees the dataset's reference solution:

```text
Problem: {problem}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{solution}
=== Reference Solution End ===

请理解上面的推理，但不要直接复制；用自己的方式推导同一个答案。
```

See [`build_student_prompt_ids()` and `build_teacher_prompt_ids()`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/opsd/01-opsd-async.py#L279-L305).

The solution acts as privileged information that changes next-token probabilities. It is not a request for the teacher to generate a new reference answer.

For a simple example, suppose the student has written:

```text
f'(x) = 6x + 2, so at x = 2 ...
```

The student predicts the next token from the question and its own prefix. Having seen the reference solution, the teacher may assign higher probability to the correct `14` and lower probability to an incorrect branch.

The extra teaching signal comes from richer context, rather than additional model parameters.

## How does OPSD differ from SFT, GRPO and ordinary OPD?

| Method | Source of training trajectories | Supervision | Teacher | Learns on student-generated states? |
| --- | --- | --- | --- | --- |
| SFT | Fixed expert trajectories | Token-level cross-entropy | No online teacher required | No |
| GRPO | Current-policy rollouts | Sequence-level reward | Reward / verifier | Yes |
| Ordinary OPD | Current-student rollouts | Teacher token log probabilities | Independent teacher model | Yes |
| OPSD | Current-student rollouts | Privileged-context teacher log probabilities | Same initial model, different prompt | Yes |

OPSD retains three useful properties:

1. **On-policy data:** training examples come from what the current student generates.
2. **Dense feedback:** every completion token receives a teacher signal.
3. **Self-distillation:** a larger teacher model is unnecessary.

## Which objective does this reproduction use?

The paper describes two implementations.

Full-vocabulary logit distillation obtains teacher and student distributions over the vocabulary at each position, then calculates JSD or KL. The main experiments use `JSD β=0.5`; later official code adds point-wise KL clipping to prevent a few style tokens from dominating.

Sampled-token distillation calculates log probabilities only for tokens actually sampled by the student, using their teacher–student difference as token advantages.

This chapter uses the second approach.

For student-sampled token `ŷ_t` at position `t`:

```text
reverse_kl_t = log p_S(ŷ_t | x, ŷ_<t>)
             - log p_T(ŷ_t | x, y*, ŷ_<t>)

advantage_t  = -β × reverse_kl_t
             = β × (log p_T - log p_S)
```

The code is two lines:

```python
reverse_kl = np.asarray(student_lps) - np.asarray(teacher_lps)
advantages = -args.kl_penalty_coef * reverse_kl
```

See [`run_prompt_rollout_async()`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/opsd/01-opsd-async.py#L373-L468).

If the teacher favors the token more than the student, `log p_T - log p_S` is positive and training encourages it. A negative difference discourages it.

## Reproduction configuration

The experiment uses:

| Item | This implementation |
| --- | --- |
| Base Model | `Qwen/Qwen3.5-4B` |
| Student | LoRA rank 64; train attention and MLP |
| Teacher | Fixed step-0 `Qwen/Qwen3.5-4B` |
| Training data | `siyanzhao/Openthoughts_math_30k_opsd` |
| Dataset size | 29,434 `problem + solution` pairs |
| Analyzed training interval | 100 steps |
| Questions per step | 32 |
| Student completions per question | 1 |
| Maximum completion length | 1,024 tokens |
| Maximum remote concurrency | 32 |
| Student / teacher thinking | Both disabled |
| Sampling | temperature 1.1 / top-p 0.95 / top-k 20 |
| KL coefficient | 1.0 |
| Learning rate | `5e-6` |
| Sampler refresh | Every step, keeping rollouts on-policy |
| Loss | `importance_sampling` |
| Checkpoint | Save state and sampler weights every 25 steps |

The run was initially configured for 200 steps, so the SwanLab name still includes `steps200`. This article analyzes only the first 100 completed, saved steps and uses 100-step reproduction commands.

## How did PyTRIO help complete this within two days?

For an introduction, see [What is PyTRIO?](https://zhuanlan.zhihu.com/p/2063265307226019219). Here I focus on the work it removed from this OPSD experiment.

The official implementation uses TRL's experimental GOLD Trainer and Accelerate, with Conda and FlashAttention in the setup and vLLM for evaluation. The paper's experiments use `8×A100`.

Starting from an ordinary development machine, that approach requires infrastructure setup:

```text
CUDA / PyTorch 版本
FlashAttention
FSDP 或其他分布式策略
vLLM / SGLang rollout
训练权重与推理权重同步
多卡资源调度
checkpoint 与恢复
```

Those frameworks are useful for teams with GPU clusters and a need to control distributed execution in detail.

My immediate goal was to investigate whether OPSD worked within two days, while keeping setup time manageable.

PyTRIO separates responsibilities as follows:

| Local Python | Remote PyTRIO service |
| --- | --- |
| Download and read data | Student sampling |
| Construct student / teacher prompts | Teacher logprob forward passes |
| `asyncio` concurrency and batching | LoRA forward / backward passes |
| Reverse KL and advantage calculation | Optimizer steps |
| SwanLab logging and analysis | State / sampler weights storage |

The local machine needs no GPU. SwanLab metadata identifies this run's host as an `Apple M4` MacBook Air; PyTRIO executes training and sampling remotely.

The first 100 `time/step_elapsed_time` values sum to about `7,540.6` seconds, or **2 hours 6 minutes**. This excludes data preparation and full AIME25 evaluation, but demonstrates the path from algorithm code to an evaluable checkpoint without maintaining a multi-GPU server locally.

For me, the setup time saved mattered as much as those training hours. I could focus on:

```text
Teacher 到底应该看到什么？
Teacher 能不能生成新 token？
Student 和 Teacher 的 logprob 如何逐 token 对齐？
prompt 区间为什么必须 mask？
什么时候刷新 Student sampler 才算严格 on-policy？
```

These are the algorithm details I wanted to understand in an OPSD reproduction.

## The asynchronous OPSD training loop

Let's inspect the actual training code.

The complete asynchronous script is:

> [`experiments/opsd/01-opsd-async.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/opsd/01-opsd-async.py)

One training step can be summarized by this pseudocode:

```python
for step in range(total_steps):
    student_sampler = refresh_latest_student_weights()

    rollouts = await asyncio.gather(
        *[
            student_sample_then_teacher_score(problem)
            for problem in batch
        ]
    )

    datums = build_importance_sampling_datums(rollouts)
    await forward_backward(datums)
    await optim_step()
```

Several alignment constraints are hidden inside that short loop.

## Step 1: prepare `problem + solution` data

The download script is [`00-datasets.py`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/opsd/00-datasets.py).

Training uses the authors' public [`siyanzhao/Openthoughts_math_30k_opsd`](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd), pinned to a revision containing `29,434` examples. Each includes at least:

```text
problem   # Student 和 Teacher 都能看到
solution  # 只有 Teacher 能看到
```

`solution` is not an SFT target here.

It conditions only the teacher distribution and is not appended to student targets. The tokens the student learns from still come from its own rollouts.

## Step 2: create a trainable student and fixed teacher

See [`train()`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/opsd/01-opsd-async.py#L558-L616):

```python
training_client = await service_client.create_lora_training_client_async(
    base_model=args.base_model,
    rank=args.lora_rank,
    train_attn=True,
    train_mlp=True,
    train_unembed=False,
)

teacher_client = await service_client.create_sampling_client_async(
    base_model=args.base_model,
)
```

Both start from `Qwen/Qwen3.5-4B`.

Their differences are:

- The student has a trainable LoRA adapter.
- The teacher receives no `model_path` and retains the step-0 base policy.
- Each step updates only the student.
- The teacher has no optimizer and does not drift with the student.

A fixed teacher avoids moving the supervision target alongside every student update.

## Step 3: sample on-policy student trajectories

At each step start, obtain a sampler for the latest student weights:

```python
student_sampler = (
    await training_client.save_weights_and_get_sampling_client_async()
)
```

The student samples using only the question:

```python
sample_result = await student_sampler.sample_async(
    prompt=student_prompt,
    num_samples=1,
    sampling_params=sampling_params,
    return_text=False,
)
```

This experiment samples one completion per question, up to 1,024 tokens.

`sampler_refresh_steps=1` refreshes the sampler after every optimizer update. Each new batch therefore comes from the latest student instead of a policy several steps behind.

That is what on-policy sampling means in this implementation.

## Step 4: score the same student completion with the teacher

After generation, the teacher receives:

```text
teacher_prompt_ids + student_completion_ids
```

See [`teacher_completion_logprobs_async()`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/opsd/01-opsd-async.py#L308-L330):

```python
all_ids = teacher_prompt_ids + completion_ids
all_logprobs = await teacher_client.compute_logprobs_async(
    trio.ModelInput.from_ints(all_ids)
)
teacher_logprobs = all_logprobs[len(teacher_prompt_ids):]
```

The call is `compute_logprobs_async()`, not `sample_async()`.

The teacher performs a teacher-forced forward pass on the student's original completion, using a prompt that includes the reference solution. It then selects the completion-position log probabilities.

Three lengths must match exactly:

```text
Student completion tokens
Student rollout logprobs
Teacher completion logprobs
```

A token mismatch invalidates the reverse-KL comparison. The code checks lengths and `None` entries instead of silently truncating.

## Step 5: convert logprob gaps into token advantages

For the same student token:

```python
reverse_kl = student_logprobs - teacher_logprobs
advantages = -kl_penalty_coef * reverse_kl
```

This is the main difference from GRPO.

GRPO derives advantages from final reward differences across rollouts for a question. OPSD derives them from teacher–student logprob differences at each token.

The supervision is therefore token-level and does not require judging final-answer correctness. In this implementation the teacher scores after the student completion finishes. Even an incorrect completion can provide a learning signal wherever teacher and student token preferences differ.

That makes the teacher's understanding of the reference solution important. Privileged context does not automatically provide reliable supervision when the model is weak or the problem too difficult.

## Step 6: train completions and mask prompts

A PyTRIO `importance_sampling` Datum requires:

```text
model_input
target_tokens
old logprobs
advantages
```

See [`build_opd_datum()`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/opsd/01-opsd-async.py#L333-L371).

The core logic is:

```python
input_ids = student_prompt_ids + completion_ids[:-1]

target_tokens = [0] * prompt_loss_len + completion_ids
old_logprobs = [0.0] * prompt_loss_len + student_logprobs
advantages = [0.0] * prompt_loss_len + token_advantages
```

Prompt-position `target_tokens`, `logprobs` and `advantages` are filled with 0. They provide context but no optimization weight.

Only student-generated completion tokens enter the loss.

The usual autoregressive shift removes the final completion token from `model_input` while retaining the full completion in `target_tokens`, so position `t` predicts the target at `t+1`.

## Step 7: run the batch concurrently with `asyncio.gather`

Each OPSD question requires at least two kinds of remote request:

1. Student sample；
2. Teacher compute logprobs。

Running all 32 questions sequentially would spend substantial time waiting for networks and remote jobs.

The asynchronous implementation preserves student → teacher dependencies within a question and runs questions concurrently:

```python
rollouts = await asyncio.gather(
    *(rollout_and_track(row) for row in batch)
)
```

One `asyncio.Semaphore(32)` bounds total student-sampling and teacher-scoring concurrency.

See [`run_prompt_rollout_async()`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/opsd/01-opsd-async.py#L373-L468) and the [batch gather in `train()`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/opsd/01-opsd-async.py#L623-L666).

Concurrency preserves the on-policy boundary: every rollout in the batch uses one student checkpoint, and the optimizer updates only after all finish.

## Step 8: one forward/backward call and a student update

After flattening all Datums for the step, submit one training update:

```python
fwd_bwd_future = await training_client.forward_backward_async(
    datums,
    loss_fn="importance_sampling",
)
optim_future = await training_client.optim_step_async(adam)

await fwd_bwd_future
await optim_future
```

See [`01-opsd-async.py#L670-L699`](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/opsd/01-opsd-async.py#L670-L699).

PyTRIO handles remote forward/backward computation, gradient accumulation and LoRA optimization. Local code controls rollout, advantages, batch boundaries and update timing.

The next step and sampler refresh begin only after the optimizer finishes.

## SwanLab training records

Trainer loss, reverse KL, advantages, token counts and time are recorded in SwanLab. Selected curves follow; [the complete run is available here](https://swanlab.cn/@kmno4/llm-agent-rl-lab-opsd/runs/pxsnuza4/chart).

![](./images/swanlab-record.png)

Across the first 100 steps:

```text
trainer/loss_mean:    0.0577 → 0.0425
reverse_kl_mean:      0.0644 → 0.0471
reverse_kl_std:       0.4378 → 0.3687
```

The gap between student and privileged teacher on student-sampled tokens decreases overall, indicating that the training objective is being optimized.

A negative `opd/advantage_mean` is expected under this definition:

```text
advantage = -(student_logprob - teacher_logprob)
```

When the student favors its own sampled tokens more than the teacher does on average, sampled reverse KL is positive and average advantage is negative.

However:

> Declining loss and reverse KL show closer agreement with the teacher; they do not directly establish improved mathematical ability.

Independent evaluation is still required. The AIME25 checkpoint curve illustrates this: the training objective declines while accuracy fluctuates.

## Running this reproduction

### 1. Install and log in

```bash
git clone https://github.com/datawhalechina/agentic-rl-lab.git
cd agentic-rl-lab

uv sync
trio login
swanlab login
```

The local host needs only a CPU environment; PyTRIO performs model sampling and LoRA training remotely.

### 2. Download OPSD and AIME25 data

```bash
uv run python experiments/opsd/00-datasets.py
```

The script downloads and validates:

```text
experiments/opsd/datasets/
├── openthoughts_math_30k_opsd/   # 29,434 条训练数据
└── aime_2025/                     # 30 道评测题
```

Both datasets use pinned Hugging Face revisions to avoid silent changes from upstream updates.

### 3. Run 100 steps of asynchronous OPSD

```bash
uv run python experiments/opsd/01-opsd-async.py \
    --steps 100 \
    --batch-size 32 \
    --group-size 1 \
    --max-tokens 1024 \
    --sample-size 0 \
    --save-every-steps 25 \
    --max-concurrency 32 \
    --swanlab-mode online
```

Every 25 steps, it saves:

```text
*-state            # 包含优化器状态，用于断点续训
*-sampler_weights  # 用于采样和 AIME25 评测
```

### 4. Evaluate the base model

```bash
uv run python experiments/opsd/00-eval-aime25.py \
    --val-n 12 \
    --max-tokens 38912 \
    --temperature 1.0 \
    --enable-thinking false \
    --output experiments/opsd/eval-results/aime25-base.jsonl
```

### 5. Evaluate an OPSD checkpoint

Insert the `trio://` path returned by `save_weights_for_sampler` in the training log:

```bash
uv run python experiments/opsd/00-eval-aime25.py \
    --val-n 12 \
    --max-tokens 38912 \
    --temperature 1.0 \
    --enable-thinking false \
    --model-path trio://<your_sampler_weights_path> \
    --output experiments/opsd/eval-results/aime25-sampler-steps100.jsonl
```

Evaluation saves 12 generations per question and writes a unified summary on the final JSONL line.

### 6. Plot the checkpoint comparison

Once `eval-results/` contains Base, Step 25, 50, 75 and 100 JSONL files:

```bash
uv run python experiments/opsd/analysis.py
```

Outputs are written to:

```text
experiments/opsd/images/aime25-opsd-progress.png
```

## Summary

OPSD reminds me of reviewing one's own work.

The student first writes its own closed-book response. The teacher then uses the reference solution to assess those original tokens, indicating which to encourage or discourage. It needs neither a larger model nor a replacement answer, only a forward pass under richer context.

We ran Qwen3.5-4B for 100 steps on the 29,434-example OpenThoughts math dataset with PyTRIO. Training cost `41.81` yuan; full AIME25 evaluation of Base plus 4 checkpoints cost another `83.61` yuan. Cumulative step time was about **2 hours 6 minutes**. At Step 100, Average@12 rises from `51.67%` to `52.78%`, and Pass@12 from `80.00%` to `86.67%`.

The improvement is modest and intermediate results fluctuate. The practical achievement for me was turning a new paper into these concrete artifacts within two days:

```text
可运行代码
真实训练
可恢复 checkpoint
独立评测
完整实验记录
```

Starting from the official multi-GPU stack might have required much of that time for CUDA, FlashAttention, rollout engines and weight synchronization. Here I spent it on prompts, token alignment, advantages and on-policy boundaries.

## References

### Paper and official implementation

1. Siyan Zhao et al. [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734), 2026.
2. [siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD)
3. Siyan Zhao. [Self-Distilled Reasoner: On-Policy Self-Distillation](https://siyan-zhao.github.io/blog/2026/opsd/)

### Data and model

1. [siyanzhao/Openthoughts_math_30k_opsd](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd)
2. [yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025)
3. [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

### This implementation

1. [Complete OPSD PyTRIO code](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/opsd)
2. [PyTRIO documentation](https://docs.pytrio.com/docs)
3. [What is PyTRIO? Introduction on Zhihu](https://zhuanlan.zhihu.com/p/2063265307226019219)
4. [PyTRIO on-policy distillation example](https://docs.pytrio.com/docs/example/opd)
5. [SwanLab record for this experiment](https://swanlab.cn/@kmno4/llm-agent-rl-lab-opsd/runs/pxsnuza4/chart)
6. [SwanLab documentation](https://docs.swanlab.cn/)
