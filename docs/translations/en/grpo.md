# Reproducing Ten RL Algorithms, Chapter 1: GRPO with Complete Code

> Want to skip the main content and get started quickly? Please refer to the [Quick Start Guide](./start.md): Prepare data → Train → Evaluate.

In my previous post, I discussed the intuition behind several common loss functions in reinforcement learning – `importance_sampling`, `ppo`, and `cispo` – and what they aim to optimize. Today, I'm officially starting a series titled "Next, I will reproduce 10 reinforcement learning papers." I'll begin with GRPO.

![](./images/GRPO.png)

<div align="center">
  <a href="https://www.zhihu.com/people/feng-qi-xia-pian" target="_blank"><img alt="Zhihu" src="https://img.shields.io/badge/Zhihu-知乎-4362f6"></a>
  <a href="https://www.xiaohongshu.com/user/profile/63c2055e000000002502c58c" target="_blank"><img alt="Rednote" src="https://img.shields.io/badge/Rednote-小红书-e93c49"></a>
  <a href="https://github.com/datawhalechina/agentic-rl-lab"><img alt="visitors" src="https://komarev.com/ghpvc/?username=KMnO4-zx-agentic-rl-lab-grpo&amp;label=visitors&amp;color=1283c3&amp;style=flat"></a>
</div>

The next chapter will discuss OPD, or On-Policy Distillation. It's quite similar to GRPO; both methods continue training using trajectories sampled from the current policy. The key difference is that GRPO uses the reward signal, while OPD uses the teacher's evaluation of the student's trajectories. We will explain this in more detail in the next chapter.

This task is very well-suited for PyTRIO. Traditionally, you could certainly set up an 8-GPU machine and manually integrate the training process, sampling, weight synchronization, logging, and checkpointing. However, what if I just want to compare the differences between three different loss functions today? Or what if I want to run experiments tomorrow with 10 different parameter combinations, including `group_size`, learning rates, and clip thresholds? At that point, the biggest cost isn't necessarily "whether you can write the GRPO code," but rather the ongoing effort required to manage the infrastructure and trainer.

Here's what makes PyTRIO so beneficial: it only handles data writing, reward calculations, loss computation, and the experimental loop locally. Forward passes, backpropagation, optimizer updates, LoRA weight saving, and the sampling service are all delegated to a remote system. This allows me to focus more on conducting experiments rather than first setting up an entire training platform.

## Where did GRPO come from?

GRPO, which stands for Group Relative Policy Optimization, was first systematically introduced in the DeepSeek-AI DeepSeekMath paper in 2024.

> DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

Paper address: https://arxiv.org/abs/2402.03300

This paper's primary focus is to continue pre-training the DeepSeek-Coder-Base-v1.5 7B model using mathematical data, resulting in DeepSeekMath. Subsequently, instruction tuning and reinforcement learning (RL) are applied. GRPO is utilized within the reinforcement learning stage to further enhance mathematical reasoning capabilities.

Why introduce GRPO? In short: PPO works well, but it's too resource-intensive.

PPO is an actor-critic method. In addition to the policy model, it also trains a value model to estimate the baseline. For LLMs, this value model is often a very large model itself, which significantly increases memory and computational requirements. Furthermore, in tasks like mathematical problem solving, scoring is often only possible based on the final answer. It's not a natural process to expect the value model to accurately estimate the value of each individual token.

The GRPO approach is very straightforward.

1. For the same question, collect a set of responses.
2. Assign a reward to each response.
3. Instead of further training the value model, use the average reward of this set of responses as the baseline.
4. Responses with a score higher than the group average will have a positive advantage; those with a score lower will have a negative advantage.
5. Use this advantage to update the policy.

![](./images/grpo-flow-xiaohei.png)

In formula notation, it's expressed as:

$$
A_i = r_i - \mathrm{mean}(r_1, r_2, \ldots, r_G)
$$

In the paper, it typically also involves dividing by the standard deviation within each group to standardize the data.

$$
A_i = \frac{r_i - \mathrm{mean}(r_1, r_2, \ldots, r_G)}{\mathrm{std}(r_1, r_2, \ldots, r_G)}
$$

This demo, in order to maintain clarity, only utilizes `reward - mean_reward`. For small experiments like GSM8K, which use a 0/1 reward system, this is already sufficient to run the core functionality of GRPO: within the same question, relatively better answers are encouraged, while relatively weaker answers are suppressed.

## What does this reproduction cover?

Code: https://github.com/datawhalechina/agentic-rl-lab

Create PyTRIO and SwanLab accounts to run the experiment.

If you clone this repository directly, the dependencies are already specified in `pyproject.toml`.

```bash
uv sync
```

If you simply want to integrate this script into your project and run it, you won't need many additional packages.

```bash
uv add "datasets>=5.0.0" "numpy>=2.5.1" "pytrio>=0.2.0" "swanlab>=0.8.4" "torch>=2.12.1" "tqdm>=4.68.3"
```

Without using `uv`, you can also proceed directly:

```bash
pip install "datasets>=5.0.0" "numpy>=2.5.1" "pytrio>=0.2.0" "swanlab>=0.8.4" "torch>=2.12.1" "tqdm>=4.68.3"
```

PyTRIO：https://pytrio.cn/

SwanLab：https://swanlab.cn/

Today, debugging this code cost me 55 yuan. If it only runs 10 steps, the cost could be as low as 5 yuan. The cost is quite reasonable. You can start by running a small batch to see if the script works, and then gradually adjust the parameters.

![](./images/trio-consume.png)

```text
experiments/grpo/01-demo-sync.py
experiments/grpo/02-demo-async.py
```

This section primarily discusses the synchronized version of `01-demo-sync.py`. The asynchronous version performs the same function, but instead submits each question within a batch concurrently, making it better suited for running formal experiments.

This demo's task is to solve mathematical problems from the GSM8K dataset. The process is:

1. Select a batch of questions from the GSM8K train split.
2. Convert each question into a chat prompt.
3. Save a sampler using the current LoRA weights.
4. Generate `group_size` answers for the same question.
5. Extract the last answer (`\boxed{...}`) from the generated responses.
6. Compare the answer with the GSM8K standard answer; award a reward of 1 if correct, and 0 if incorrect.
7. Calculate the advantage within each group of questions.
8. Construct a PyTRIO `Datum`.
9. Update the model using either `importance_sampling`, `ppo`, or `cispo`.
10. Log metrics to SwanLab and save the LoRA weights.

We can start with a lower-cost version.

```bash
trio login

uv run python experiments/grpo/01-demo-sync.py \
  --steps 10 \
  --batch-size 4 \
  --group-size 8 \
  --max-tokens 512 \
  --loss-fn importance_sampling \
  --swanlab-mode online
```

If you're just checking whether the script runs successfully, you can temporarily disable SwanLab.

```bash
uv run python experiments/grpo/01-demo-sync.py \
  --steps 1 \
  --batch-size 2 \
  --group-size 2 \
  --max-tokens 64 \
  --loss-fn importance_sampling \
  --swanlab-mode disabled
```

![alt text](./images/terminal-run.png)

## Step 1: Prepare the prompt and reward.

GRPO does not specify how rewards should be written. It only focuses on one thing: scoring multiple responses to the same prompt and then comparing them within the group.

In this code, the prompt instructs the model to write the final answer within the "`\boxed{}`" field.

```python
QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."
```

To ensure the model responds more consistently with the desired format, a small few-shot example has been added to the beginning of the code.

```python
FEWSHOT_PREFIX = [
    {"role": "user", "content": "How many r's are in strawberry?" + QUESTION_SUFFIX},
    {
        "role": "assistant",
        "content": (
            "<think>\n\n</think>\n\n"
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. "
            "There are three r's. \\boxed{3}"
        ),
    },
]
```

The core function that builds the prompt is quite short.

```python
def build_prompt(tokenizer: Any, question: str) -> list[int]:
    messages = [
        *FEWSHOT_PREFIX,
        {"role": "user", "content": question + QUESTION_SUFFIX},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not prompt_tokens:
        raise ValueError("Prompt tokens are empty")
    return prompt_tokens
```

The reward was also deliberately kept very simple. It only accepted the final code "`\boxed{...}`"; it awarded a score of 1 if it matched the standard answer, and 0 otherwise.

```python
def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    return matches[-1].strip()


def grade_answer(response: str, ground_truth: str) -> float:
    answer = extract_boxed(response)
    if answer is None:
        return 0.0
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0
```

This is one of the most appealing aspects of RLVR, or Reinforcement Learning with Verifiable Rewards: it doesn't require you to train a reward model first. Whether it's math problems, coding challenges, format validation, or improving the success rate of tool calls, as long as you can write a verifiable scoring function, you can start running the reinforcement learning loop.

## Step 2: Sample a set of responses using the current policy.

GRPO's sampling process doesn't generate just one answer per question. It must generate multiple completions for the same prompt. The advantage lies in comparing relative values within the group, and with only one response, comparison is not possible.

In the synchronized version, the core function is `run_rollout_group`.

```python
def run_rollout_group(
    sampling_client: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    ground_truth: str,
    sampling_params: trio.SamplingParams,
    group_size: int,
) -> list[RolloutSample]:
    result = sampling_client.sample(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=group_size,
        sampling_params=sampling_params,
        return_text=True,
    ).result()
```

There are two details that are particularly important.

First, the sampling method uses `sampling_client`, not `training_client`. In each step, the code first saves a sampler from the current LoRA weights.

```python
sampling_client = training_client.save_weights_and_get_sampling_client()
```

This step means that the current sampler represents the "old policy." The `logprobs` used when constructing the loss function must come from the sampling results obtained during that "old policy." It's crucial not to recalculate it after the model is updated – we can't simply fudge the results.

Secondly, `num_samples=group_size` indicates that a single question allows for multiple answers to be submitted. Once the results are obtained, the code will save three things:

```python
tokens = list(sequence.tokens)
logprobs = [float(value) for value in sequence.logprobs]
reward = grade_answer(text, ground_truth)
```

`tokens` contains the completion tokens generated by the model.

`logprobs` represents the log probability of these tokens during the old policy sampling process.

`reward` is the score we generated locally.

These three items, combined, represent the raw materials needed for the upcoming training.

## Step 3: Calculating group-relative advantage.

Once scores have been assigned to all answers for a given question, the advantage is then calculated.

```python
mean_reward = sum(rewards) / len(rewards)
return [
    RolloutSample(
        tokens=tokens,
        logprobs=logprobs,
        text=text,
        reward=reward,
        advantage=reward - mean_reward,
    )
    for (tokens, logprobs, text), reward in zip(raw_samples, rewards, strict=True)
]
```

For example, let's assume a question where we are sampling 4 responses, and the reward is:

```text
[1, 0, 1, 0]
```

Within the group, the average reward is `0.5`. Therefore, the advantage is:

```text
[+0.5, -0.5, +0.5, -0.5]
```

This is the core of GRPO: it's not simply about getting the right answer to earn points, but rather about how your performance compares to the average within a particular set of responses to the same question.

If all the rewards in a particular group are the same, for example:

```text
[0, 0, 0, 0]
```

Alternatively:

```text
[1, 1, 1, 1]
```

Then all advantages are zero, and this group has no training signal. The code will directly skip this section.

```python
if all(sample.advantage == 0.0 for sample in rollout_samples):
    n_degenerate += 1
    continue
```

This is also why I recommend recording `frac_degenerate`. The higher this value, the fewer questions within that batch are actually providing reliable quality signals to GRPO. When the model is too weak, it often produces entirely incorrect results, and when the model is too strong, it may also consistently produce correct results. Both of these scenarios are not ideal for GRPO.

## Step 4: Convert the rollout into a PyTRIO Datum.

PyTRIO's built-in `importance_sampling` and `ppo` losses require three types of input:

```text
target_tokens
logprobs
advantages
```

They must all be precisely the same length as `model_input`.

The code performs this action within `build_grpo_datum`.

```python
def build_grpo_datum(prompt_tokens: list[int], sample: RolloutSample) -> trio.Datum:
    if not sample.tokens:
        raise ValueError("Cannot train on an empty completion")

    observation_len = len(prompt_tokens) - 1
    input_tokens = prompt_tokens + sample.tokens[:-1]
    target_tokens = [0] * observation_len + sample.tokens
    padded_logprobs = [0.0] * observation_len + sample.logprobs
    padded_advantages = [0.0] * observation_len + [sample.advantage] * len(sample.tokens)

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
            "advantages": np.asarray(padded_advantages, dtype=np.float32),
        },
    )
```

The most common mistake here is the autoregressive token shift.

Language model training always involves predicting the next token based on the preceding tokens. Therefore:

```text
model_input = prompt + completion[:-1]
target      =          completion
```

What about the prompt section? The prompt serves as context, not as the object we'll be applying reinforcement learning to. The code uses a placeholder value of 0.

```python
target_tokens = [0] * observation_len + sample.tokens
padded_logprobs = [0.0] * observation_len + sample.logprobs
padded_advantages = [0.0] * observation_len + [sample.advantage] * len(sample.tokens)
```

Only completion tokens carry non-zero advantages; prompt tokens do not contribute to the policy-gradient loss.

This is extremely important. GRPO isn't training the model to simply "memorize" prompts, but rather to train the model so that, after seeing a prompt, it is more likely to generate completions that lead to higher rewards.

## Step 5: Submit forward/backward pass and optimizer step.

Once a batch of data has been collected, the training process is actually quite short.

```python
if config.loss_fn in BUILTIN_LOSS_FNS:
    fwd_bwd_future = training_client.forward_backward(
        datums,
        loss_fn=config.loss_fn,
    )
```

`BUILTIN_LOSS_FNS` is:

```python
BUILTIN_LOSS_FNS = {"importance_sampling", "ppo"}
```

In other words, both `importance_sampling` and `ppo` can directly reuse the same set of GRPO data. The difference lies not in the rollout process, but rather in how losses are handled for `logprobs` and `advantages`.

Then, proceed with the optimizer update.

```python
optim_future = training_client.optim_step(adam_params)
fwd_bwd_result = fwd_bwd_future.result()
optim_future.result()
loss_metrics = dict(fwd_bwd_result.metrics)
```

The local code simply submits the training tasks to PyTRIO. The actual model forward pass, backpropagation, and LoRA parameter updates are all performed remotely. For me, this is why PyTRIO is so well-suited for these reproduction experiments: it allows me to focus on the algorithmic variables, such as reward, advantage, loss, and group size, rather than constantly dealing with the training infrastructure.

## What do the three different loss functions optimize?

A key feature of this demo is that the same GRPO rollout can switch between three different loss functions.

```bash
--loss-fn importance_sampling
--loss-fn ppo
--loss-fn cispo
```

They are using the same data.

```text
target_tokens: 模型实际生成的 token
logprobs:      old policy 采样这些 token 时的 logprob
advantages:    这条 completion 的组内相对好坏
```

The difference lies in how to convert these elements into gradients.

### 1. Importance sampling: Directly boosting the probability of desirable tokens while suppressing undesirable ones.

The intuition behind Importance Sampling is:

```text
ratio = exp(current_logprob - old_logprob)
objective = ratio * advantage
```

If the advantage is positive, it indicates that this completion is better than the average within its group. Therefore, increase the probability of the current model generating these tokens.

If the advantage is negative, it indicates that this completion is worse than the average within the group. Therefore, reduce the probability of the current model generating these tokens.

Here, `ratio` is used to correct the situation where "the data is derived from the old policy, but we are currently training the current policy." The greater the difference between the old policy and the current policy, the more the ratio deviates from 1.

Its advantages are straightforward, and so are its disadvantages: without a clip, the strategy might be updated quite rapidly. This is convenient for small experiments, as we can clearly see how the reward signal feeds into the loss; however, more caution is typically needed during formal training.

### 2. PPO: The objective remains the same, but we've added a speed limit to policy updates.

PPO is still doing that.

```text
让高 advantage token 概率上去，让低 advantage token 概率下来
```

However, it will apply a clipping mechanism to the ratio. Intuitively, if the current policy already significantly favors a particular token compared to the old policy, there's no need to continue pushing it forward with the original intensity.

You can think of PPO as importance sampling with a speed limit.

```text
unclipped = ratio * advantage
clipped   = clip(ratio, 1 - eps, 1 + eps) * advantage
objective = min(unclipped, clipped)
```

Therefore, the objective of PPO optimization is to maximize the weighted generated probability, while ensuring that the current policy does not deviate significantly from the previous policy.

This is also why many RLHF/RLVR systems tend to favor PPO or PPO variants. It's not the simplest option, but it generally offers better stability.

### 3. cispo: Treat the clipped ratio as a fixed weight, and truly optimize the log probability.

This demo implements `cispo` as a custom PyTRIO loss. Its core formula is:

```python
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
clipped_ratio = torch.clamp(
    prob_ratio,
    min=clip_low_threshold,
    max=clip_high_threshold,
)
cispo_objective = clipped_ratio.detach() * target_logprobs * advantages
loss = -cispo_objective.sum()
```

Its key difference from PPO lies in `.detach()`.

The term `clipped_ratio.detach()` refers to a factor that can influence the weight of this particular loss term, but it no longer participates in backpropagation. What is actually being optimized is `target_logprobs`.

Therefore, this is how I understand CISPO:

```text
PPO:   clip objective，让目标本身更保守。
CISPO: clip ratio，把它当成固定砝码，控制每个 token 的学习力度。
```

Considering these three together:

| Loss Function | Optimization Goal | Stabilization Method | Suitable for Observing |
|---|---|---|---|
| `importance_sampling` | Directly maximize `ratio * advantage` | No apparent rate limiting | How the reward signal most directly influences token probabilities. |
| `ppo` | Maximize the clipped surrogate objective | The clip ratio limits policy updates. | Given the same reward, does a more conservative update provide greater stability? |
| `cispo` | Maximize `detach(clipped_ratio) * logprob * advantage` | After ratio clipping, it functions as a fixed weight. | How can a custom loss function be used to control the shape of the gradient? |

Please note that here, we are comparing loss objectives, not stating that one is definitively superior. Experiments like GRPO are highly sensitive to factors such as reward design, data difficulty, group size, sampling temperature, learning rate, and the number of training steps. It is precisely because of these many variables that the value of PyTRIO's parallel experiments becomes apparent.

## How to implement custom loss functions using PyTRIO?

This is the part that I find most interesting.

If only the built-in `importance_sampling` or `ppo` are being used, we will directly provide the datums to:

```python
training_client.forward_backward(datums, loss_fn=config.loss_fn)
```

However, `cispo` does not utilize the built-in loss function; instead, it uses:

```python
training_client.forward_backward_custom(custom_datums, loss_fn)
```

The division of labor for PyTRIO custom loss is:

1. The remote model is responsible for the forward pass, calculating the log probability for each token with respect to `target_tokens`.
2. The `loss_fn` module, implemented in local Python, receives these differentiable log probabilities.
3. You can write any loss function using PyTorch, and it should return `(loss, metrics)`.
4. PyTRIO performs the backward pass and parameter updates based on this loss.

That means you're not training the model locally; you're simply defining locally how those log probabilities should be transformed into a loss function.

### First, transform the Datum into the format required for the custom forward process.

The `importance_sampling` data includes:

```text
target_tokens
logprobs
advantages
```

The custom forward pass only needs the current policy to recompute log probabilities for `target_tokens`, so the code first converts the Datum:

```python
def build_custom_forward_datum(datum: trio.Datum) -> trio.Datum:
    return trio.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={
            "target_tokens": datum.loss_fn_inputs["target_tokens"],
        },
    )
```

The log probabilities and advantage values from the old policy cannot be lost. They are passed into the custom loss function through a closure.

```python
sampling_logprobs_list = [
    get_float_tensor_values(datum, "logprobs") for datum in datums
]
advantages_list = [
    get_float_tensor_values(datum, "advantages") for datum in datums
]
```

### Then write the loss function

`make_cispo_loss_fn` returns the `cispo_loss_fn` callback passed to PyTRIO.

```python
def make_cispo_loss_fn(
    sampling_logprobs_list: list[list[float]],
    advantages_list: list[list[float]],
    clip_low_threshold: float,
    clip_high_threshold: float,
):
    def cispo_loss_fn(data, logprobs_list):
        datum_losses = []

        for target_logprobs, sampling_values, advantage_values in zip(
            logprobs_list,
            sampling_logprobs_list,
            advantages_list,
            strict=True,
        ):
            target_logprobs = target_logprobs.float()
            device = target_logprobs.device
            sampling_logprobs = torch.as_tensor(
                sampling_values,
                dtype=torch.float32,
                device=device,
            )
            advantages = torch.as_tensor(
                advantage_values,
                dtype=torch.float32,
                device=device,
            )

            prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
            clipped_ratio = torch.clamp(
                prob_ratio,
                min=clip_low_threshold,
                max=clip_high_threshold,
            )
            cispo_objective = clipped_ratio.detach() * target_logprobs * advantages
            datum_losses.append(-cispo_objective.sum())

        loss = torch.stack(datum_losses).sum()
        return loss, {"loss_mean": float(loss.detach().item())}

    return cispo_loss_fn
```

In the actual code, several more metrics are also tracked, such as:

```text
cispo/train_tokens
cispo/ratio_mean
cispo/clipped_ratio_mean
cispo/clip_fraction
```

These metrics are quite helpful. For example, a high `clip_fraction` value indicates that a significant proportion of tokens are being clipped. In this situation, you should examine whether the learning rate is too high or if the current sampling strategy deviates too far from the intended strategy.

Finally, submit the call:

```python
custom_datums = [build_custom_forward_datum(datum) for datum in datums]
fwd_bwd_future = training_client.forward_backward_custom(
    custom_datums,
    make_cispo_loss_fn(
        sampling_logprobs_list=sampling_logprobs_list,
        advantages_list=advantages_list,
        clip_low_threshold=config.cispo_clip_low_threshold,
        clip_high_threshold=config.cispo_clip_high_threshold,
    ),
)
```

This approach has significance beyond simply addressing CISPO. In the future, whether aiming to reproduce DAPO, GSPO, Dr.GRPO, or experimenting with a new token-level weighting method, they all represent essentially the same path.

```text
保留 rollout / reward / advantage 管线
替换 custom loss
用 SwanLab 对比实验
```

## What metrics should be tracked?

The code records only a limited number of metrics for SwanLab, but it's sufficient.

```python
log_payload = {
    "reward": mean_reward,
    "frac_degenerate": frac_degenerate,
    "rollout/avg_gen_len": avg_gen_len,
    "datums": len(datums),
    "train_tokens": train_tokens,
    ...
}
```

Please pay close attention to these few points.

| Indicator | Meaning |
|---|---|
| `reward` | Average reward per question group within the current batch. |
| `frac_degenerate` | Proportion of question groups with identical rewards and no training signal. |
| `rollout/avg_gen_len` | Average generated length. |
| `datums` | Number of completions actually used for training. |
| `train_tokens` | Number of advantage tokens with non-zero values. |
| `loss_mean` | Average loss returned by the trainer. |
| `cispo/clip_fraction` | Fraction of token ratios clipped by CISPO. |

Looking solely at the reward isn't enough. For example, if the reward hasn't changed, it could mean the training wasn't effective. It could also be that the value of `frac_degenerate` is too high, resulting in very few useful samples. Alternatively, it's possible that the model is beginning to generate longer sequences of reasoning, but the impact on the reward hasn't been reflected yet.

![alt text](./images/swanlab.png)

## How do you compare the three losses?

The simplest approach is to keep all other parameters unchanged and only replace the `--loss-fn`.

```bash
uv run python experiments/grpo/01-demo-sync.py \
  --steps 30 \
  --batch-size 4 \
  --group-size 8 \
  --max-tokens 512 \
  --loss-fn importance_sampling \
  --swanlab-mode online

uv run python experiments/grpo/01-demo-sync.py \
  --steps 30 \
  --batch-size 4 \
  --group-size 8 \
  --max-tokens 512 \
  --loss-fn ppo \
  --swanlab-mode online

uv run python experiments/grpo/01-demo-sync.py \
  --steps 30 \
  --batch-size 4 \
  --group-size 8 \
  --max-tokens 512 \
  --loss-fn cispo \
  --swanlab-mode online
```

If we want to be more precise, we can add a few more parameters.

```text
group_size: 4 / 8
learning_rate: 1e-5 / 4e-5 / 8e-5
temperature: 0.7 / 1.0
cispo_clip_high_threshold: 2.0 / 4.0
```

This is the PyTRIO benefit mentioned earlier. Many frameworks can run one GRPO demo; organizing 3 losses, 10 configurations and all the records, weights and logs takes more work. With training resources managed by PyTRIO, local experiments become ordinary Python jobs. Run configurations in separate processes or with a simple batch launcher, then compare the SwanLab results.

Of course, this doesn't mean the resources are free. Sampling and training still consume computational power on the remote server. Its value lies in the fact that you don't need to maintain your own 8-GPU training service, nor do you need to reprocess the combined training and inference pipeline, weight synchronization, and checkpoint management every time you modify the loss function.

## What are the differences between this demo version and the paper version of GRPO?

To ensure the first section is clear, I have kept a few simplifications.

First, the reward system is rule-based, operating on a 0/1 reward structure. The DeepSeekMath paper also discusses various configurations, including reward models, outcome supervision, and process supervision. This paper will begin with a verifiable reward system, as it is the easiest to reproduce and is most suitable for mathematical problems.

Second, the demo calculates advantages using only:

```text
reward - mean_reward
```

The paper will include intra-group standardization. If you want to align it more closely with the paper, you could consider changing it to:

```python
std_reward = np.std(rewards)
advantage = (reward - mean_reward) / (std_reward + 1e-8)
```

Third, this code does not explicitly include a reference-policy KL term. Our primary focus here is to clearly explain the rollout, advantages, and loss functions for GRPO, as well as the training pipeline for PyTRIO. To achieve a more complete reproduction of the research, you could add constraints related to the reference policy within the reward function or custom loss.

These simplifications do not affect the core objective of the first section: to successfully execute the training logic for GRPO and to analyze the 'loss' variable in isolation.

## Conclusion

The key to GRPO isn't a "mysterious new RL algorithm," but rather a very simple substitution.

```text
PPO:  用 value model 估 baseline
GRPO: 用同一 prompt 下多个回答的组内平均分估 baseline
```

This makes it particularly well-suited for tasks that can be verified, such as math problems, coding challenges, and tasks involving tool usage. As long as you can define a reward function, you can first generate a set of responses to the same prompt, calculate the relative advantage within that set, and then update the model using a policy gradient-based loss function.

This article corresponds to the PyTRIO code and performs three actions.

1. Run a minimal GRPO using the GSM8K + boxed answer reward evaluation method.
2. Compare `importance_sampling`, `ppo`, and `cispo` using the same rollout dataset.
3. Demonstrate how to implement your own loss function within PyTRIO using `forward_backward_custom`.

The next chapter, OPD, will replace the reward system with teacher signals. Instead of asking "Is this answer correct?", we will ask "Does the teacher approve of the trajectory the student has taken?" This approach will be closer to distillation and better suited for tasks where there isn't a standard answer but where a strong teacher is available.
