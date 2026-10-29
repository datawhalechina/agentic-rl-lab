# Reproducing Ten RL Algorithms, Chapter 2: Improving Medical Knowledge with OPD While Preserving General Capabilities

> Ready to run the code? See the [quick start](./start.md): prepare data → train → evaluate.

![](./images/OPD-head.png)

<div align="center">
  <a href="https://www.zhihu.com/people/feng-qi-xia-pian" target="_blank"><img alt="Zhihu" src="https://img.shields.io/badge/Zhihu-知乎-4362f6"></a>
  <a href="https://www.xiaohongshu.com/user/profile/63c2055e000000002502c58c" target="_blank"><img alt="Rednote" src="https://img.shields.io/badge/Rednote-小红书-e93c49"></a>
  <a href="https://github.com/KMnO4-zx/agentic-rl-lab"><img alt="visitors" src="https://komarev.com/ghpvc/?username=KMnO4-zx-agentic-rl-lab-medical-opd&amp;label=visitors&amp;color=1283c3&amp;style=flat"></a>
</div>

> **Code and reproduction resources**
>
> - Open-source repository: [KMnO4-zx/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab)
> - PyTRIO website and registration: [https://pytrio.cn/](https://pytrio.cn/) (remote training, sampling and weight saving)
> - SwanLab registration: [https://swanlab.cn/login](https://swanlab.cn/login) (training logs and experiment metrics)
> - Medical SFT dataset: [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
> - MedQA-zh evaluation data: [bigbio/med_qa](https://huggingface.co/datasets/bigbio/med_qa)
> - C-Eval dataset: [ceval/ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam)
> - Prerequisite: [General OPD tutorial](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/general-opd/readme.md)
> - PyTRIO Skill：[SwanHubX/pytrio-skill](https://github.com/SwanHubX/pytrio-skill)
> - SwanLab Skill：[SwanHubX/swanlab-skill](https://github.com/SwanHubX/swanlab-skill)

The day after deciding to write this series, I had already finished the code and tutorial for DeepMath OPD. It was a clean introductory experiment: the student samples a math solution, the teacher computes log probabilities for that same trajectory, and reverse KL supplies the student update. I could have published that as the next chapter.

But a working script alone felt too small a contribution to the series and to the readers waiting for it. I wanted each chapter to leave a question worth discussing, together with a reasonably complete account of the experiment.

So I spent about another week designing experiments from Medical SFT onward: training a medical teacher, trying Medical OPD, and gradually addressing the loss of general capabilities after domain training. That led to the two approaches in this post and MedQA-zh and C-Eval evaluations at every stage. I learned a lot, including from the failures, and hope these notes help others exploring OPD, distillation and domain adaptation. All of my experiments together cost roughly 500 yuan. At the time, I estimated that reproducing the selected runs could cost under 200 yuan, the DeepMath OPD experiment about 50 yuan, and a few dozen exploratory steps just a few yuan. These are historical estimates, not fixed prices or guaranteed budgets.

I enjoy Lilian Weng's blog at Thinking Machines Lab. Reading her posts feels like listening to someone explain why a question mattered, what they tried, what happened, and how it changed their judgment. I hope to get closer to that style over time.

Why use PyTRIO for these reproductions? I had tried open-source RL frameworks such as verl, but found their setup and architecture heavy for experiments focused on loss functions. I also tried Tinker and PyTRIO. In my experience at the time, PyTRIO's Alipay payment option was more convenient from China than the foreign-currency credit card I needed for Tinker. Another consideration was utilization: Agentic RL combines generation, tools and training, and long rollouts can leave rented GPUs waiting. PyTRIO's token-based billing model made that idle time less of a concern for these experiments.

> Which would you like to see next: GSPO or Search-R1?

In the previous GRPO chapter, reward supplied the training signal. We sampled several responses to one question and increased the probability of trajectories that scored above the group average, while decreasing the probability of weaker ones.

OPD uses a different signal. It does not require a reference answer for every prompt or a hand-written reward function. The student generates its own response, and the teacher scores each token along that actual trajectory. The update then moves the student toward the distribution preferred by the teacher.

The DeepMath-103K version is available as a [General OPD tutorial](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/general-opd/readme.md), covering the minimal training loop. Here I wanted to find out whether that mechanism could solve a more practical problem.

A friend's suggestion led me to Chinese medical capability enhancement.

I first fine-tuned `Qwen/Qwen3.5-4B` on `medical-o1-reasoning-SFT-zh`. Medical evaluation improved, but C-Eval accuracy fell from `81.67%` to `69.33%`. The model gained domain knowledge while losing a substantial amount of its original general capability.

That raised the question for this chapter:

> Can a medical teacher teach domain knowledge while the original base model anchors general capabilities, producing a student that improves on the base model in both areas?

I tested two approaches:

1. **SAR-OPD (Staged Anchor-Restoration OPD):** run Medical OPD first, then restore general capabilities in a separate stage with the base teacher.
2. **IDT-OPD (Interleaved Dual-Teacher OPD):** alternate medical-teacher and base-teacher steps within one training loop for the same student.

The main result: in this experiment, the final SAR-OPD checkpoint reached `84.33%` on MedQA-zh and `84.67%` on C-Eval non-med, exceeding the original 4B base model on both.

> SAR-OPD and IDT-OPD are names I use for these two experimental designs. They are not established algorithm names from earlier papers. This chapter implements the core OPD mechanism and explores how it can balance medical and general capabilities.

All code is here: <https://github.com/KMnO4-zx/agentic-rl-lab>

## Results First

### A consistent evaluation protocol

I used two fixed test sets to compare the different stages:

| Capability | Evaluation set | Samples | Output budget | Metric |
| --- | --- | ---: | ---: | --- |
| Medical | MedQA-zh 4 options | 600 | 1024 tokens | Accuracy |
| General | C-Eval non-med | 300 | 8192 tokens | Accuracy |

Here `1024` and `8192` are maximum output tokens per question. The tables below report the actual sample counts separately from those generation budgets.

### Key checkpoints from both approaches

| Model / Checkpoint | MedQA-zh (600 questions) | C-Eval non-med (300 questions) |
| --- | ---: | ---: |
| 4B Base | 72.17% `433/600` | 81.67% `245/300` |
| Medical SFT epoch 3 | 79.00% `474/600` | 69.33% `208/300` |
| Medical OPD step 300 | 80.00% `480/600` | 73.00% `219/300` |
| **SAR-OPD step 300** | **84.33% `506/600`** | **84.67% `254/300`** |
| IDT-OPD step 100 | 79.17% `475/600` | 81.67% `245/300` |
| IDT-OPD step 200 | 76.50% `459/600` | **86.00% `258/300`** |

Before going into the training designs, take a look at the complete results for each approach.

**Approach 1: SAR-OPD.** The restoration trajectory on the right is the key: Base-anchor OPD gradually brings C-Eval back, while MedQA-zh remains above the base model and eventually improves further. Both final metrics exceed the original 4B base.

![](./images/sar_opd_analysis.png)

**Approach 2: IDT-OPD.** Look at the Pareto frontier on the right. Step 100 favors medical capability; step 200 favors general capability. Continuing to step 600 does not improve the balance, so the final checkpoint is not the best one.

![](./images/idt_opd_analysis.png)

Three observations stand out.

First, Medical SFT raises medical accuracy from `72.17%` to `79.00%`, while C-Eval loses `12.34` percentage points. Looking only at the domain benchmark would hide that tradeoff.

Second, SAR-OPD ultimately reaches `84.33%` on MedQA-zh, `12.17` percentage points above the base model, and `84.67%` on C-Eval, an increase of `3.00` percentage points. This is the strongest combined checkpoint in this experiment.

Third, IDT-OPD does not end at a point where every metric keeps improving. Step 100 favors medical capability and step 200 favors general capability, forming a tradeoff curve. With multiple teachers, checkpoint selection and early stopping matter more than simply reaching the last scheduled step.

## What Does OPD Train?

OPD stands for On-Policy Distillation. Here, on-policy means that training trajectories come from the current student, rather than a fixed response dataset or answers generated by the teacher in advance.

Two lines of work motivate this setup. [Generalized Knowledge Distillation (GKD)](https://arxiv.org/abs/2306.13649), introduced in 2023, emphasizes teacher feedback on student-generated sequences. [MiniLLM](https://arxiv.org/abs/2306.08543) systematically studies reverse KL for generative language-model distillation. This implementation combines student-generated trajectories with teacher scoring of those same trajectories and reverse-KL advantages for the student update.

A minimal OPD iteration works as follows:

1. Take a question from a prompt-only dataset.
2. Sample a completion from the current student and retain its sampling-time token log probabilities.
3. Pass the exact same `prompt + student completion` to the teacher.
4. The teacher computes log probabilities for those tokens, without generating another answer.
5. Construct reverse-KL advantages from the student and teacher log-probability difference.
6. Update the student with PPO or importance sampling.
7. Export the updated student sampler for the next step.

![](./images/opd-training-loop.png)

Let the student be $\pi_\theta$, the teacher $\pi_T$, and the student's sampled token at position $t$ be $y_t$. The code uses this token-level reverse-KL sample:

```math
r_t^{\mathrm{KL}}
=
\log \pi_\theta(y_t \mid x,y_{\lt t})
-
\log \pi_T(y_t \mid x,y_{\lt t}).
```

It is converted into an advantage:

```math
A_t=-\beta r_t^{\mathrm{KL}}.
```

The corresponding code is just two lines:

```python
reverse_kl = np.asarray(student_logprobs) - np.asarray(teacher_logprobs)
advantages = -args.kl_penalty_coef * reverse_kl
```

When the student assigns a token more probability than the teacher does, `reverse_kl` is positive and the advantage is negative, pushing that token's probability down. When the teacher prefers the token more strongly, the positive advantage pushes its probability up.

Compare that with the previous GRPO chapter:

```text
GRPO: advantage 来自同一组回答的相对 reward
OPD:  advantage 来自 Teacher 与 Student 的 token logprob 差
```

OPD also has a `group_size`, which simply samples more student trajectories per prompt. It does not compute group-relative rewards as GRPO does; each completion independently receives token-level teacher supervision.

If OPD is new to you, start with the [General OPD tutorial](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/general-opd/readme.md). It explains reverse KL, autoregressive shifting, Datum alignment, and synchronous and asynchronous training in more detail. This chapter focuses on the design and results of the two medical experiments.

## Preparing the Data

The experiment uses three types of data:

| Dataset | Purpose | Fields used during training |
| --- | --- | --- |
| Chinese split of `FreedomIntelligence/medical-o1-reasoning-SFT` | Medical SFT and Medical OPD prompt pool | SFT uses `question + complex_cot + response`; OPD uses only `question` |
| MedQA-zh 4 options from `bigbio/med_qa` | Medical evaluation | Held-out evaluation only |
| Eight selected non-medical subsets of `ceval/ceval-exam` | Base-anchor OPD and general evaluation | OPD uses questions and options, ignoring `answer_idx` |

The download script prepares data with a fixed seed:

```bash
uv run python experiments/medical-opd/00-download-dataset.py
```

C-Eval is first split `80% / 20%` into an OPD training pool and a held-out test pool; 300 fixed questions are then sampled from the latter. To reduce medical overlap, the selection excludes `basic_medicine`, `clinical_medicine`, `physician`, `veterinary_medicine`, high-school biology and middle-school biology. The eight retained non-medical subsets are:

```text
computer_network
college_programming
advanced_mathematics
discrete_mathematics
college_physics
logic
chinese_language_and_literature
college_economics
```

The data boundaries are deliberate:

- None of the 600 MedQA-zh test questions enters training.
- The held-out C-Eval test pool is excluded from OPD.
- Although the C-Eval OPD training pool includes `answer_idx`, the training script ignores it and uses only questions and options as prompts.
- Medical OPD reads `question` only, ignoring the original `complex_cot` and `response` fields.
- The teacher scores the student's actual completion; it does not generate training answers.

Both OPD approaches therefore use prompt-only distillation. Supervision comes from the teacher's distribution rather than dataset answers.

## Step One: Build a Medical Teacher

Both approaches share the same medical teacher.

I used `02-medical-sft.py` to train `Qwen/Qwen3.5-4B` on the Chinese `medical-o1-reasoning-SFT` data. The source separates reasoning and the final answer into `complex_cot` and `response`; the script assembles them as follows:

```text
<think>
{complex_cot}
</think>

{response}
```

During SFT, system and user prompts provide context with zero loss weight. Only the assistant completion and EOS receive nonzero weights:

```python
tokens = prompt_tokens + assistant_tokens
weights = [0.0] * len(prompt_tokens) + [1.0] * len(assistant_tokens)

datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": np.asarray(tokens[1:], dtype=np.int32),
        "weights": np.asarray(weights[1:], dtype=np.float32),
    },
)
```

I trained for 3 epochs, saving a full training state and inference sampler weights after each epoch. The evaluation results were:

| Checkpoint | MedQA-zh (600 questions) | C-Eval non-med (300 questions) |
| --- | ---: | ---: |
| 4B Base | 72.17% `433/600` | 81.67% `245/300` |
| Medical SFT epoch 1 | 77.67% `466/600` | 71.67% `215/300` |
| Medical SFT epoch 2 | **79.17% `475/600`** | 69.33% `208/300` |
| Medical SFT epoch 3 | 79.00% `474/600` | 69.33% `208/300` |

Medical capability improved, but C-Eval fell from `81.67%` to `69.33%`. I selected the checkpoint after all 3 epochs as a fixed teacher for both OPD experiments. It is frozen to provide medical token-level supervision to fresh students, rather than used directly as the final model.

Two saved artifact types serve different purposes:

| Path type | Purpose |
| --- | --- |
| `trio://.../sampler_weights/...` | Inference, evaluation and frozen-teacher use |
| `trio://.../training_states/...` | Restore a TrainingClient and continue training |

Medical OPD needs the SFT `sampler_weights` for its teacher. SAR-OPD's third stage needs the Medical OPD `training_state` to resume its student. These paths are not interchangeable.

## Approach One: SAR-OPD, Staged Anchor Restoration

The first design lets the student focus on medical capability, then uses the original base model to restore general capabilities.

I call it Staged Anchor-Restoration OPD, or SAR-OPD.

![](./images/SAR-OPD.png)

The full workflow has three stages:

```text
Stage 1: Qwen3.5-4B --Medical SFT--> Medical Teacher

Stage 2: Fresh 4B Student --Medical Teacher OPD--> Medical OPD Student

Stage 3: Medical OPD Student --4B Base Teacher OPD--> SAR-OPD Student
```

### Stage 1：Medical SFT

Stage one is the teacher preparation already described. The Medical SFT epoch-3 checkpoint is frozen and only computes log probabilities for the student's medical completions.

### Stage 2: Medical OPD with a fresh student

Stage two creates a fresh LoRA student from `Qwen/Qwen3.5-4B`, rather than continuing from the SFT weights:

```python
service_client = trio.ServiceClient()

training_client = await service_client.create_lora_training_client_async(
    base_model=args.base_model,
    rank=args.lora_rank,
    seed=args.seed,
)

teacher_client = await service_client.create_sampling_client_async(
    base_model=args.teacher_base_model,
    model_path=args.teacher_model_path,
)
```

Here `training_client` is the trainable student, while `teacher_client` points to the Medical SFT epoch-3 sampler weights.

At the beginning of each step, export a sampler from the current student weights:

```python
student_sampler = (
    await training_client.save_weights_and_get_sampling_client_async()
)
```

The student then samples `group_size` completions for each medical question:

```python
sample_result = await student_sampler.sample_async(
    prompt=trio.ModelInput.from_ints(prompt_ids),
    num_samples=args.group_size,
    sampling_params=sampling_params,
    return_text=False,
)
```

The teacher receives the same `prompt + completion`:

```python
all_ids = prompt_ids + completion_ids
all_logprobs = await teacher_client.compute_logprobs_async(
    trio.ModelInput.from_ints(all_ids)
)
teacher_logprobs = all_logprobs[len(prompt_ids):]
```

The teacher must not generate a separate answer here. OPD compares student and teacher probabilities for the exact same tokens in the exact same contexts. Different completions would break that token-level alignment.

Finally, construct reverse-KL advantages and apply a PPO update:

```python
reverse_kl = np.asarray(student_logprobs) - np.asarray(teacher_logprobs)
advantages = -args.kl_penalty_coef * reverse_kl

fwd_bwd_future = await training_client.forward_backward_async(
    datums,
    loss_fn="ppo",
)
optim_future = await training_client.optim_step_async(adam)

fwd_bwd_result = await fwd_bwd_future
await optim_future
```

The main Medical OPD run used this configuration:

```text
Student: Qwen/Qwen3.5-4B fresh LoRA
Teacher: Medical SFT epoch 3
Steps: 300
Batch size: 4 prompts
Group size: 4 completions per prompt
Max completion tokens: 2048
Learning rate: 4e-5
Loss: PPO
Sampler refresh: every step
```

After 300 steps, medical accuracy rose from the base model's `72.17%` to `80.00%`, but C-Eval reached only `73.00%`. That improved on Medical SFT's `69.33%`, while still falling short of the base model's general capability.

I continued with a third stage.

### Stage 3: Restore general capabilities with the base teacher

Stage three restores the complete Medical OPD step-300 training state into a new TrainingClient:

```python
training_client = await service_client.create_training_client_from_state_async(
    path=args.student_state_path
)
```

The original `Qwen/Qwen3.5-4B` becomes the frozen teacher:

```python
teacher_client = await service_client.create_sampling_client_async(
    base_model="Qwen/Qwen3.5-4B"
)
```

Training prompts now contain C-Eval non-med questions and options, without reading `answer_idx`. The student still generates its own answers, and the base teacher scores those actual completions.

The objective is to move the student back toward the original base model's token distribution on general questions. The base model serves as an anchor, hence the name Base-anchor C-Eval OPD.

Stage three uses a more conservative configuration than Medical OPD:

```text
Student initialization: Medical OPD step 300 Train state
Teacher: original Qwen/Qwen3.5-4B
Steps: 300
Batch size: 4 prompts
Group size: 4 completions per prompt
Max completion tokens: 2048
Learning rate: 5e-6
Loss: PPO
Checkpoint: every 50 steps
```

I reduced the learning rate from `4e-5` to `5e-6` because this stage is closer to restoration and fine adjustment. I wanted to avoid erasing the medical capability just acquired.

### SAR-OPD results

![](./images/sar_opd_analysis.png)

The full results are:

| Stage / Checkpoint | MedQA-zh (600 questions) | C-Eval non-med (300 questions) |
| --- | ---: | ---: |
| 4B Base | 72.17% `433/600` | 81.67% `245/300` |
| Medical SFT epoch 1 | 77.67% `466/600` | 71.67% `215/300` |
| Medical SFT epoch 2 | 79.17% `475/600` | 69.33% `208/300` |
| Medical SFT epoch 3（Medical Teacher） | 79.00% `474/600` | 69.33% `208/300` |
| Medical OPD step 300 (stage-three step 0) | 80.00% `480/600` | 73.00% `219/300` |
| Base-anchor OPD step 50 | 84.17% `505/600` | 76.67% `230/300` |
| Base-anchor OPD step 100 | 82.50% `495/600` | 79.33% `238/300` |
| Base-anchor OPD step 150 | 83.00% `498/600` | 80.33% `241/300` |
| Base-anchor OPD step 200 | 82.17% `493/600` | 83.67% `251/300` |
| Base-anchor OPD step 250 | 83.33% `500/600` | 82.67% `248/300` |
| **Base-anchor OPD step 300** | **84.33% `506/600`** | **84.67% `254/300`** |

C-Eval shows a clear recovery trend, rising from Medical OPD's `73.00%` to `84.67%`. There are intermediate fluctuations, but it ultimately passes the base model's `81.67%`.

The MedQA-zh result was more surprising. The Base-anchor stage uses no medical prompts, and its teacher is the original base model, yet medical accuracy rises from `80.00%` to `84.33%`. In this run, restoring the general distribution and preserving medical capability did not form a strictly zero-sum tradeoff.

This remains one training result, not a general rule. Learning rate, prompts, generation budgets and checkpoint variance could all affect the numbers. Multiple seeds and ablations are needed to establish what caused the gain.

The SAR-OPD checkpoint I selected was:

```text
trio://run_1vbwh91n5le5/sampler_weights/opd-ceval-async-qwen35-4b-base-anchor-ppo-steps300-step000300
```

Relative to the original 4B base model:

```text
MedQA-zh:       72.17% -> 84.33%  (+12.17 percentage points)
C-Eval non-med: 81.67% -> 84.67%  (+3.00 percentage points)
```

## Approach Two: IDT-OPD, Interleaved Dual Teachers

SAR-OPD performed well, but required a longer engineering workflow:

```text
训练 SFT Teacher
    ↓
训练 Medical OPD Student
    ↓
保存并恢复 Train state
    ↓
继续做 Base-anchor OPD
```

I wanted to try another design: could one fresh student receive supervision from both teachers within a single training loop?

I call this Interleaved Dual-Teacher OPD, or IDT-OPD.

![](./images/IDT-OPD.png)

It uses the same trained Medical SFT teacher, with a fresh student initialized from `Qwen/Qwen3.5-4B`. Training follows a fixed `1:1` schedule:

```text
Medical -> C-Eval -> Medical -> C-Eval -> ...
```

Medical steps use the Medical SFT epoch-3 teacher; C-Eval steps use the original 4B base teacher. Both task types share:

- One LoRA student;
- One TrainingClient;
- One Adam optimizer state;
- One global step counter;
- The latest student weights after every update.

### Implementing the fixed 1:1 schedule

The code first constructs a cycle from the configuration:

```python
def build_task_schedule(args):
    return (
        ("medical",) * args.medical_steps_per_cycle
        + ("ceval",) * args.ceval_steps_per_cycle
    )
```

When both parameters equal 1, the schedule is:

```python
("medical", "ceval")
```

The training loop uses the global step to select the task, data and teacher:

```python
task = schedule[step % len(schedule)]

if task == "medical":
    rows = medical_rows
    teacher_client = medical_teacher
    system_message = args.medical_system_message
else:
    rows = ceval_rows
    teacher_client = base_teacher
    system_message = args.ceval_system_message
```

Student sampling, teacher log probabilities, reverse KL and PPO updates then follow the same logic. Only the prompts and scoring teacher change for the current step.

I also maintain a separate task-step counter for each task:

```python
task_steps = {"medical": 0, "ceval": 0}

batch = batch_for_task_step(
    rows=rows,
    task_step=task_steps[task],
    batch_size=args.batch_size,
)

task_steps[task] += 1
```

This lets Medical and C-Eval advance through their own batches without skipping half the data because the global steps alternate.

### IDT-OPD configuration

```text
Student: Qwen/Qwen3.5-4B fresh LoRA
Medical Teacher: Medical SFT epoch 3
C-Eval Teacher: original Qwen/Qwen3.5-4B
Schedule: Medical 1 step + C-Eval 1 step
Global steps: 600 = M300 + C300
Batch size: 4 prompts
Group size: 4 completions per prompt
Max completion tokens: 2048
Learning rate: 1e-5
Loss: PPO
Sampler refresh: every global step
```

Every 100 global steps contain 50 medical optimizer steps and 50 C-Eval optimizer steps. Thus step 200 means `M100 + C100`, rather than 200 steps for each task.

### IDT-OPD results

![](./images/idt_opd_analysis.png)

| Checkpoint | Task progress | MedQA-zh (600 questions) | C-Eval non-med (300 questions) |
| --- | --- | ---: | ---: |
| 4B Base | - | 72.17% `433/600` | 81.67% `245/300` |
| Medical SFT Teacher | - | 79.00% `474/600` | 69.33% `208/300` |
| **IDT-OPD step 100** | `M50 + C50` | **79.17% `475/600`** | 81.67% `245/300` |
| **IDT-OPD step 200** | `M100 + C100` | 76.50% `459/600` | **86.00% `258/300`** |
| IDT-OPD step 300 | `M150 + C150` | 76.33% `458/600` | 84.33% `253/300` |
| IDT-OPD step 400 | `M200 + C200` | 75.17% `451/600` | 81.33% `244/300` |
| IDT-OPD step 500 | `M250 + C250` | 75.33% `452/600` | 83.67% `251/300` |
| IDT-OPD step 600 | `M300 + C300` | 75.00% `450/600` | 82.00% `246/300` |

Step 100 favors medical capability: MedQA-zh reaches `79.17%`, while C-Eval exactly preserves the base model's `81.67%`.

Step 200 favors general capability: MedQA-zh is `76.50%`, while C-Eval reaches the IDT run's maximum of `86.00%`. Both exceed the base model by `4.33` percentage points, giving a more balanced checkpoint.

Continuing to step 600 does not improve the result. Medical accuracy declines to `75.00%`, and C-Eval falls back to `82.00%`. A fixed 1:1 schedule does not automatically converge to a better balance; teacher interaction, learning rate, task order and training duration all shape the trajectory.

The useful IDT-OPD result is therefore the pair of Pareto-efficient checkpoints at steps 100 and 200:

```text
step 100: 更偏医疗，同时把 C-Eval 保持在 Base 水平
step 200: 医疗仍高于 Base，C-Eval 达到最高
```

For a more balanced overall result, I would choose step 200:

## Choosing Between the Two Approaches

| Dimension | SAR-OPD | IDT-OPD |
| --- | --- | --- |
| Training structure | Sequential stages | Alternate teachers at each step |
| Student initialization | Fresh base for Medical OPD; resume its state for stage three | Fresh base student throughout one training loop |
| Teacher use | Medical teacher, followed by base teacher | Medical and base teachers alternate |
| Best MedQA-zh in this experiment | **84.33%** | 79.17% |
| Best C-Eval in this experiment | 84.67% | **86.00%** |
| Main advantage | Strongest combined result here, with an explicit recovery stage | Unified structure that can extend task ratios and teacher counts |
| Main difficulty | Longer workflow with correct training-state saving and restoration | Sensitive to scheduling, learning rate, early stopping and checkpoint selection |
| Suitable exploration | The strongest combined final performance observed in this run | Multiple domains and teachers, continued training and Pareto tradeoffs |

Based on this experiment alone, I would select SAR-OPD stage-three step 300 as the final model. It exceeds the base on both MedQA-zh and C-Eval and has the strongest combined result across the two approaches.

IDT-OPD is interesting for its structure. A medical teacher and a base teacher could be extended to teachers for code, math or agent tool use, with scheduling ratios determining which capabilities receive more training attention.

More teachers also introduce complexity: their tokenizers must be compatible, task prompts must align, scheduling ratios require tuning, and the shared optimizer state combines update histories from different tasks. IDT-OPD offers flexibility, without an inherent guarantee of greater stability than sequential training.

## Running the Experiments

After cloning the repository, install dependencies and log in to PyTRIO:

```bash
git clone https://github.com/KMnO4-zx/agentic-rl-lab.git
cd agentic-rl-lab

uv sync
trio login
```

Training scripts use the PyTRIO SDK. The two evaluation scripts use TRIO's OpenAI-compatible API and need the same API key in `experiments/medical-opd/.env`:

```bash
cp experiments/medical-opd/.env.example experiments/medical-opd/.env
```

Replace the placeholder in `.env`:

```text
PYTRIO_API_KEY="your_pytrio_api_key_here"
```

### 1. Download and prepare the data

For the full download:

```bash
uv run python experiments/medical-opd/00-download-dataset.py
```

To check the data pipeline with a small run first:

```bash
uv run python experiments/medical-opd/00-download-dataset.py \
  --medical-sft-sample-size 100 \
  --medqa-sample-size 20 \
  --ceval-train-size 100 \
  --ceval-test-size 20
```

### 2. Train the Medical SFT teacher

A small trial run:

```bash
uv run python experiments/medical-opd/02-medical-sft.py \
  --sample-size 100 \
  --num-epochs 1 \
  --batch-size 2 \
  --max-length 2048 \
  --swanlab-mode disabled
```

The full training run:

```bash
uv run python experiments/medical-opd/02-medical-sft.py \
  --num-epochs 3 \
  --batch-size 16 \
  --max-length 2048 \
  --swanlab-mode online
```

After training, record the epoch-3 `sampler_weights` path. Both approaches use it as the medical teacher.

### 3. Run Medical OPD

```bash
uv run python experiments/medical-opd/03-medical-opd-async.py \
  --teacher-model-path YOUR_SFT_SAMPLER_WEIGHTS_PATH \
  --steps 300 \
  --batch-size 4 \
  --group-size 4 \
  --sample-size 0 \
  --max-tokens 2048 \
  --learning-rate 4e-5 \
  --save-every-steps 300 \
  --swanlab-mode online
```

The completed run saves both:

```text
Medical OPD Train state      -> 用于 SAR-OPD 第三阶段继续训练
Medical OPD sampler weights  -> 用于评测或推理
```

### 4. Run SAR-OPD's Base-anchor stage

```bash
uv run python experiments/medical-opd/04-ceval-opd-async.py \
  --student-state-path YOUR_MEDICAL_OPD_TRAIN_STATE_PATH \
  --steps 300 \
  --batch-size 4 \
  --group-size 4 \
  --sample-size 0 \
  --max-tokens 2048 \
  --learning-rate 5e-6 \
  --save-every-steps 50 \
  --swanlab-mode online
```

`--student-state-path` must point to a training state produced by `save_state`, not a `/sampler_weights/` path. Before any C-Eval OPD update, the script saves a source-copy state. It also saves step-0 state and sampler weights for the main run so the recovery trajectory can be compared precisely.

### 5. Run IDT-OPD

```bash
uv run python experiments/medical-opd/05-interleaved-multi-teacher-opd.py \
  --medical-teacher-model-path YOUR_SFT_SAMPLER_WEIGHTS_PATH \
  --steps 600 \
  --medical-steps-per-cycle 1 \
  --ceval-steps-per-cycle 1 \
  --batch-size 4 \
  --group-size 4 \
  --max-tokens 2048 \
  --learning-rate 1e-5 \
  --save-every-steps 50 \
  --swanlab-mode online
```

To try a `2:1` schedule favoring medical steps, change:

```bash
--medical-steps-per-cycle 2 \
--ceval-steps-per-cycle 1
```

Prefer saving checkpoints at the end of complete scheduling cycles. A `1:1` cycle has length 2, so use a save interval divisible by 2. A `2:1` cycle has length 3, so use a save interval divisible by 3.

### 6. Evaluate sampler weights

MedQA-zh：

```bash
uv run python experiments/medical-opd/01-eval-medical.py \
  --model YOUR_SAMPLER_WEIGHTS_PATH \
  --max-tokens 1024 \
  --concurrency 16
```

C-Eval non-med：

```bash
uv run python experiments/medical-opd/01-eval-ceval.py \
  --model YOUR_SAMPLER_WEIGHTS_PATH \
  --max-tokens 8192 \
  --concurrency 16
```

Evaluation writes per-question JSONL and summary metrics JSON files to `experiments/medical-opd/eval-results/`. Keep the data files, system prompt, temperature and output budgets identical when comparing checkpoints; otherwise the accuracies are not directly comparable.

## Metrics to Watch During Training

Both OPD scripts log to SwanLab. Useful metrics include:

| Metric | Meaning |
| --- | --- |
| `opd/reverse_kl_mean` | Mean student-minus-teacher log probability on sampled tokens |
| `opd/reverse_kl_std` | Dispersion of token-level reverse-KL samples |
| `data/datums` | Completions actually included in the current training step |
| `data/completion_tokens_mean` | Average tokens per completion |
| `data/completion_tokens_total` | Total completion tokens for the current step |
| `step/completion_tokens_per_second` | Token throughput across the entire OPD step |
| `trainer/*` | PPO loss and other training metrics returned by PyTRIO |

IDT-OPD records task-specific metrics under `medical/*` and `ceval/*`, along with:

```text
train/medical_steps
train/ceval_steps
train/medical_ratio
train/ceval_ratio
```

The tokens/s metric includes student sampling, teacher log probabilities, forward/backward passes, optimizer updates and sampler refresh. It measures the whole OPD step, not inference-only generation speed.

A decrease in sampled reverse KL suggests closer agreement with the current teacher; it does not guarantee a benchmark gain. IDT-OPD has two teachers with different objectives, so loss or KL alone cannot establish the final capability balance. Save checkpoints regularly and evaluate them on independent held-out data.

## Limits of This Experiment

Both workflows ran end to end with fixed evaluation sets, but this is not a complete paper-level ablation study.

First, each approach has one main training trajectory, without multi-seed means or variances. Results such as `84.33%` may include checkpoint fluctuations; small differences should not all be read as stable improvements.

Second, the learning rates differ: Medical OPD uses `4e-5`, Base-anchor OPD uses `5e-6`, and IDT-OPD uses `1e-5`. Differences between the approaches cannot therefore be attributed entirely to sequential versus interleaved training.

Third, I have not systematically ablated `group_size`, `kl_penalty_coef`, PPO versus importance sampling, sampler-refresh intervals, or the Medical/C-Eval scheduling ratio. The completed `1:1` IDT-OPD run does not establish the optimal ratio.

Fourth, MedQA-zh and C-Eval are multiple-choice benchmarks. They help track capability changes but do not establish safety in real clinical conversations, and their accuracies do not show that a model is suitable for medical diagnosis.

Finally, this chapter implements the OPD training loop and explores two applications. It does not strictly reproduce every dataset, model and benchmark from GKD, MiniLLM or another OPD paper. The supported conclusion is narrower: both training approaches ran with this code, these configurations and fixed evaluations, producing checkpoint results that can be checked.

## Closing Notes

The experiment began as an attempt to add Chinese medical knowledge to a 4B model. The central problem became balancing domain improvement against general-capability forgetting.

Medical SFT taught domain knowledge quickly, but C-Eval fell from `81.67%` to `69.33%`. OPD provided another training mechanism: the student first exposes what it currently generates, then different teachers supply token-level supervision along those actual trajectories.

SAR-OPD separates two training objectives: learn medical capability, then restore general capability with the base anchor. It achieved the strongest combined result in this experiment:

```text
MedQA-zh:       84.33%
C-Eval non-med: 84.67%
```

IDT-OPD puts both teachers in one loop. It did not produce a monotonically improving final point, but yielded two Pareto checkpoints with different emphases and demonstrated a flexible structure for scheduling capabilities in multi-teacher OPD.

That is what I find most interesting about OPD. Its uses extend beyond compressing a larger model into a smaller one. When teacher and student log probabilities can be aligned on the same token trajectory, different teachers can represent different capabilities, and we can choose when and on which prompts the student learns from each.

## References

### Methods and models

1. Rishabh Agarwal et al. [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649), 2023.
2. Yuxian Gu, Li Dong, Furu Wei, Minlie Huang. [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543), 2023.
3. Junying Chen et al. [HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs](https://arxiv.org/abs/2412.18925), 2024.

### Datasets and evaluation benchmarks

1. Di Jin et al. [What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams](https://arxiv.org/abs/2009.13081), 2020.
2. Yuzhen Huang et al. [C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models](https://arxiv.org/abs/2305.08322), 2023.
3. Medical SFT dataset: [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
4. MedQA evaluation data: [bigbio/med_qa](https://huggingface.co/datasets/bigbio/med_qa)
5. C-Eval evaluation data: [ceval/ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam)

### Implementation and experiment tools

1. Complete code for this chapter: [KMnO4-zx/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/experiments/medical-opd)
2. PyTRIO documentation: [Quick start](https://docs.pytrio.cn/docs) · [Compute Logprobs](https://docs.pytrio.cn/docs/advanced/compute_logprobs) · [Custom Loss Functions](https://docs.pytrio.cn/docs/guide/loss_fn)
3. SwanLab documentation: [Quick start](https://docs.swanlab.cn/guide_cloud/general/quick-start.html)
