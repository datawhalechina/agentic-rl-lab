# Reproducing Ten RL Algorithms, Chapter 8: Teaching an LLM Household Tasks in 80 Steps

> Ready to run the code? See the [quick start](./start.md): install the environment → download data → train → evaluate.

![ALFWorld household-agent training](./images/封面.png)

<div align="center">
  <a href="https://www.zhihu.com/people/feng-qi-xia-pian" target="_blank"><img alt="Zhihu" src="https://img.shields.io/badge/Zhihu-知乎-4362f6"></a>
  <a href="https://www.xiaohongshu.com/user/profile/63c2055e000000002502c58c" target="_blank"><img alt="Rednote" src="https://img.shields.io/badge/Rednote-小红书-e93c49"></a>
  <a href="https://github.com/KMnO4-zx/agentic-rl-lab"><img alt="visitors" src="https://komarev.com/ghpvc/?username=KMnO4-zx-agentic-rl-lab-alfworld&amp;label=visitors&amp;color=1283c3&amp;style=flat"></a>
</div>

> **Code and reproduction resources**
>
> - Complete code: [KMnO4-zx/agentic-rl-lab/experiments/alfworld](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/experiments/alfworld)
> - Paper: [ALFWorld: Aligning Text and Embodied Environments for Interactive Learning](https://arxiv.org/abs/2010.03768)
> - Official implementation: [alfworld/alfworld](https://github.com/alfworld/alfworld)
> - SwanLab: [Complete ALFWorld training records](https://swanlab.cn/@kmno4/llm-agent-rl-lab-alfworld/runs)
> - PyTRIO documentation: [https://docs.pytrio.com/docs](https://docs.pytrio.com/docs)

This is the eighth chapter in my series reproducing ten reinforcement learning algorithms.

Search-R1 taught the model when to search; ReTool taught it when to write Python. Here I wanted to extend Agentic RL to **a text kitchen with real persistent state, where the model finds objects, opens cabinets, picks things up, heats or cleans them, and puts them in the right place.**

This experiment differs from the earlier ones:

- A trajectory interacts with the environment up to **50 times**.
- Evaluation trajectories average **21–23 steps**.
- Every tool call **actually changes TextWorld / PDDL state**.
- The model sees **the complete tool history from the start of the task** before its next action.
- The full training-sequence budget increases from the earlier 4K–8K range to **12K tokens**.
- Each update contains **8 games**, with **8 independent trajectories per game**, totaling **64 long trajectories**.

The cost increases accordingly. Earlier experiments could be compared with the price of a drink or two coffees; this full PyTRIO training run cost **¥1613.71**.

The run produced a measurable improvement. After 80 steps, overall success on 274 fixed games rises from `52.92%` to `56.93%`, Valid Unseen from `52.99%` to `58.96%`, and mean invalid actions fall from `5.41` to `4.81`.

![](./images/alfworld_checkpoint_evaluation.png)

This article explains ALFWorld, its tool wrapper, GRPO-style group-relative learning for long agent trajectories, and the code organization.

## 0. What is ALFWorld?

ALFWorld is the interactive environment introduced by Mohit Shridhar and colleagues in the ICLR 2021 paper [*ALFWorld: Aligning Text and Embodied Environments for Interactive Learning*](https://openreview.net/forum?id=0IOX0YcCdTn). It aligns two settings:

- **TextWorld:** agents use textual observations and actions, with PDDL maintaining internal state.
- **ALFRED / AI2-THOR:** agents receive visual input in 3D rooms and complete embodied tasks through navigation and object manipulation.

![Aligned TextWorld and AI2-THOR environments in ALFWorld](./images/alfworld_textworld_embodied.png)

*Two views of the same task family: the left shows the TextWorld interaction used here; the right shows an embodied kitchen and low-level actions in AI2-THOR. Source: [ALFWorld project](https://github.com/alfworld/alfworld/blob/master/media/alfworld_teaser.png).*

The original paper aims to learn high-level policies in an abstract text world and map them to actions in a visual environment. Consider this task:

```text
Put a heated apple in the fridge.
```

A person can naturally break it into steps:

```text
找到苹果
→ 拿起苹果
→ 找到微波炉
→ 打开微波炉
→ 放入并加热苹果
→ 取出苹果
→ 找到冰箱
→ 打开冰箱
→ 把苹果放进去
```

Think of ALFWorld as a text-based household simulator. When the model says “take the apple” or “open the refrigerator,” the simulator checks whether the action is possible. It retains changes such as the apple being held or the refrigerator being open, then reports the new situation for the next decision.

This chapter uses **text-only mode**. The model sees no kitchen image, but still interacts with a stateful environment that checks action preconditions and determines task completion.

### Six household task types

The code uses six ALFWorld task types:

| Task type | Required behavior |
| --- | --- |
| `pick_and_place_simple` | Pick up an object and place it in the specified receptacle |
| `look_at_obj_in_light` | Pick up an object and examine it under a light |
| `pick_clean_then_place_in_recep` | Clean an object before placing it in the specified receptacle |
| `pick_heat_then_place_in_recep` | Heat an object before placing it in the specified receptacle |
| `pick_cool_then_place_in_recep` | Cool an object before placing it in the specified receptacle |
| `pick_two_obj_and_place` | Find two objects of the same kind and place them in the specified receptacle |

Following [`data.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/data.py), only `game.tw-pddl` files marked `solvable=true` and belonging to these six task types are retained. The downloaded data yields:

| Split | Available games | Meaning |
| --- | ---: | --- |
| `train` | 3,553 | Training games |
| `valid_seen` | 140 | Room layouts seen during training, with different task instances |
| `valid_unseen` | 134 | Room layouts absent from training, emphasizing environmental generalization |

Seen and unseen refer to room layouts. Both splits use the same six task types, so `valid_unseen` tests transfer of learned action strategies to new rooms.

### Scope of this reproduction

The ALFWorld paper and this chapter's training method have different scopes.

The paper's main contribution aligns text and embodied environments; official agent training mainly uses methods such as [DAgger](https://proceedings.mlr.press/v15/ross11a.html). Here we use its text-only environment, add **[GRPO-style group rollouts and relative advantages](https://arxiv.org/abs/2402.03300)**, and update online with **[PPO](https://arxiv.org/abs/1707.06347)**.

The scope is an LLM Agentic RL recipe on ALFWorld. The paper's original training algorithm and visual BUTLER system are outside this reproduction.

Next, we connect the textual environment to a tool protocol an LLM can call consistently.

## 1. Turn household actions into a tool call

The model receives one tool:

```python
alfworld_step(action: str)
```

`action` uses ALFWorld's textual commands directly, for example:

```text
go to countertop 1
open fridge 1
take apple 1 from countertop 1
heat apple 1 with microwave 1
move apple 1 to fridge 1
```

`apple 1` and `fridge 1` are complete instance names. Numeric suffixes distinguish objects in the same scene and must be copied exactly.

One interaction round looks like this:

```text
任务 + 当前 observation + 可执行动作
                ↓
模型调用 alfworld_step(action)
                ↓
TextWorld / PDDL 执行动作并更新状态
                ↓
返回新的 observation + 可执行动作 + done / won
                ↓
把本轮 assistant tool call 和 tool result 追加到完整历史
```

The complete process is:

![ALFWorld text-tool interaction](<./images/ALFWorld · Tool Interaction.png>)

The environment, rather than the model's own declaration, determines termination:

- `won=True`: all task conditions are satisfied.
- `done=True, won=False`: the environment ends or reaches the interaction limit.
- A trajectory reaches 12K tokens: stop early to respect the sequence budget.

The next prompt retains the complete interaction record. After 30 turns, the 31st generation still sees all 30 actual assistant tool calls and tool observations.

Household tasks depend heavily on state. The model must remember which cabinet it opened, whether it holds the apple or placed it in the microwave, and where it moved it after heating. Losing early history can cause repeated searches, wrong-object choices and forgotten intermediate steps.

With trajectory generation defined, we can run the system before examining its rewards and advantages.

## 2. Start training and evaluation

The project requires Python `>=3.13`. ALFWorld is an optional dependency, so other chapters using ordinary `uv sync` do not need to download TextWorld.

### Install dependencies and download ALFWorld data

```bash
git clone https://github.com/KMnO4-zx/agentic-rl-lab.git
cd agentic-rl-lab

uv sync --extra alfworld
trio login
swanlab login

cd experiments/alfworld
uv run --extra alfworld alfworld-download \
    --data-dir "$PWD/datasets/alfworld"
```

Data defaults to `experiments/alfworld/datasets/alfworld`; training and evaluation need no extra `--data-root` argument.

### Launch 80-step training

```bash
uv run --extra alfworld python train.py \
    --base-model Qwen/Qwen3.5-4B \
    --max-steps 80 \
    --games-per-batch 8 \
    --group-size 8 \
    --max-episode-steps 50 \
    --max-trajectory-tokens 12000 \
    --max-assistant-tokens 2048 \
    --temperature 1.0 \
    --learning-rate 1e-6 \
    --save-every 20 \
    --swanlab-mode online
```

Every 20 updates, training saves:

- `state`: the complete training state.
- `sampler weights`: weights loaded by `eval.py` for checkpoint evaluation.

### Evaluate the base model

```bash
uv run --extra alfworld python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --output eval_results/base-qwen35-4b-eval.jsonl \
    --swanlab-mode disabled
```

### Evaluate the Step 80 checkpoint

This command uses sampler weights from the documented experiment. For your own run, replace `--model-path` with the printed `Saved sampler weights:` path.

```bash
uv run --extra alfworld python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path 'trio://run_se1v45e6nrxd/sampler_weights/alfworld-agent-rl-qwen35-4b-update-80-weights' \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --output eval_results/checkpoint-80steps.jsonl \
    --swanlab-mode disabled
```

Evaluation writes complete messages, actions, environment feedback, termination reasons and a final summary to JSONL. If the output exists, choose a new filename or deliberately pass `--overwrite-output`.

Generate the checkpoint comparison figure:

```bash
cd ..
uv run python experiments/alfworld/analysis.py
```

These commands run the experiment. Understanding what the model learns from 64 long trajectories requires examining rewards, advantages and PPO updates within each rollout batch.

## 3. How do we use the GRPO idea?

GRPO samples several responses to one problem and calculates advantages from their relative rewards.

For ALFWorld, a group becomes **8 independent environment trajectories for the same game and initial state**.

```text
同一个 game.tw-pddl
        ↓
创建 8 个相互独立的 TextWorld 环境
        ↓
Qwen3.5-4B 分别完成 8 条完整工具轨迹
        ↓
环境给出每条轨迹的终局结果
        ↓
在这 8 条轨迹内部计算相对 advantage
```

The complete training loop is:

![ALFWorld and PyTRIO training loop](<./images/ALFWorld × PyTRIO · Training Loop.png>)

### 1. One episode return per trajectory

Success starts at `1` and failure at `0`. Every invalid tool format or non-executable environment action subtracts `0.1`:

```text
reward[i] = (1 if won[i] else 0) - 0.1 * invalid_action_count[i]
```

For example:

```text
成功，并出现 2 次非法动作：reward = 1 - 0.2 = 0.8
失败，并出现 3 次非法动作：reward = 0 - 0.3 = -0.3
```

The code computes no separate step rewards, step returns, step advantages or additional action advantages. Native environment `score` is recorded only in evaluation trajectories; training reward follows the formula above.

### 2. Subtract the mean within each game group

After the 8 trajectories for a game finish, calculate:

```text
advantage[i] = reward[i] - mean(reward_of_same_game)
```

There is no division by standard deviation here.

Trajectories above the group mean have positive advantage; those below it have negative advantage. If all 8 rewards match, every advantage is 0 and the group skips remote backward computation.

### 3. Assign advantages only to model-generated tokens

A training sequence contains four kinds of content:

```text
system / user prompt
assistant reasoning + tool call
tool observation
下一轮 assistant reasoning + tool call
...
```

Only assistant-generated completion tokens are optimized:

```text
assistant completion token → trajectory advantage
system / user / tool token  → 0
```

Tool observations must remain in context because later actions depend on feedback. They are environment outputs and must not be trained as model actions.

### 4. Update LoRA with PPO

Each full trajectory becomes a PyTRIO `Datum`, retaining rollout old log probabilities and using the built-in PPO loss:

```python
training_client.forward_backward(
    datums,
    loss_fn="ppo",
    loss_fn_config={
        "clip_low_threshold": 0.8,
        "clip_high_threshold": 1.2,
    },
).result()
```

The precise description is:

> **We combine same-game GRPO-style group rollouts and relative advantages with PPO updates to Qwen3.5-4B LoRA.**

These advantages apply to a sequence of decisions that continuously changes the environment, the main difference from the earlier experiments.

## 4. A more stateful agent experiment

What excited me here was extending Agentic RL to sustained multi-turn decisions that alter real environment state.

Each action changes what can happen next:

- The refrigerator must be opened before the apple can go inside.
- Holding an object may prevent picking up another.
- Heating the apple is insufficient if it is then placed in the wrong location.
- Wrong navigation changes subsequent observations and available actions.
- Success depends on the final environment state, regardless of how plausible the earlier written plan sounds.

Evaluation averages about 22 interactions per game. Every generation sees the growing full history. This longer credit-assignment problem resembles sustained agent work more closely than one or two tool calls.

Text-only interaction keeps the experiment practical. TextWorld / PDDL supplies exact state transitions, PyTRIO handles remote sampling and training, and local Python handles environments, rewards, advantages and trajectory records.

Longer decision chains also increase context length, sampling volume and cost.

## 5. Training configuration and cost

The full experiment uses:

| Item | Configuration |
| --- | --- |
| Base Model | `Qwen/Qwen3.5-4B` |
| LoRA rank | 32 |
| Training duration | 80 updates |
| Games per update | 8 |
| Independent trajectories per game | 8 |
| Rollouts per update | 64 |
| Total rollouts | 5,120 |
| Maximum environment interactions | 50 steps / trajectory |
| Full trajectory limit | 12,000 tokens |
| Assistant generation limit per turn | 2,048 tokens |
| Sampling | temperature 1.0 / top-p 1.0 |
| Reward | `1[won] - 0.1 × invalid_actions` |
| Advantage | Same-game `reward - mean(reward)` |
| Loss | PPO，clip `0.8 / 1.2` |
| Optimizer | Adam，lr `1e-6`，β `(0.9, 0.95)` |
| Checkpoint | Save state and sampler weights every 20 updates |

### Why was this run expensive?

The PyTRIO session cost **¥1613.71**:

![](./images/pytrio-consume.png)

The token breakdown explains much of the cost:

```text
prefilling: 843.51M
train:       62.24M
sample:      14.28M
```

Prefilling greatly exceeds train and sample tokens. After every tool action, sampling receives a longer task history, including tool calls and observations. By twenty or thirty turns, the early context has been processed repeatedly.

Training tokens are only part of long-context Agentic RL cost. Rereading an expanding environment history also consumes substantial computation.

The complete run is recorded in [SwanLab](https://swanlab.cn/@kmno4/llm-agent-rl-lab-alfworld/runs). Reward and trainer metrics follow:

![](./images/swanlab-reward.png)

![](./images/swanlab-trainer.png)

Online success and rewards fluctuate with task type and difficulty across batches. Trainer loss and token counts remain within manageable ranges. Capability changes still require independent evaluation on fixed games and temperature.

## 6. Evaluation results

Base, Step 40 and Step 80 share exactly these settings:

```text
Base Model: Qwen/Qwen3.5-4B
valid_seen: 140 games
valid_unseen: 134 games
total: 274 games
temperature: 0.01
top_p: 1.0
seed: 42
max environment steps: 50
```

The complete result figure is:

![](./images/alfworld_checkpoint_evaluation.png)

| Model | Valid Seen | Valid Unseen | Overall success | Successful games | Mean reward | Mean invalid actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 52.86% | 52.99% | 52.92% | 145 / 274 | -0.0117 | 5.41 |
| Checkpoint 40 | **57.14%** | 52.24% | 54.74% | 150 / 274 | 0.0237 | 5.24 |
| Checkpoint 80 | 55.00% | **58.96%** | **56.93%** | **156 / 274** | **0.0880** | **4.81** |

Checkpoint 80 compared with Base:

```text
总体成功率:       52.92% → 56.93%   +4.01 pp
Valid Unseen:    52.99% → 58.96%   +5.97 pp
成功游戏数:       145 → 156          +11 games
平均非法动作数:    5.41 → 4.81        -0.60
```

Three observations stand out.

First, much of the gain comes from `valid_unseen`: nearly 6 percentage points on unseen room layouts. In this run, training changes transfer beyond the rooms encountered during training.

Second, Checkpoint 40 has the best `valid_seen` result, while Checkpoint 80 leads on `valid_unseen` and overall success. Improvements are not monotonic across every dimension; checkpoint selection depends on the target split.

Third, invalid actions decline. Each incurs a `0.1` reward penalty, and the model reduces both malformed and non-executable actions alongside improving task completion.

Gains differ substantially across the six task types:

![Results across six ALFWorld task types](<./images/ALFWorld · Task-Level Results.png>)

On `valid_unseen`, Heat then place gains `26.1 pp` and Clean then place gains `9.7 pp`, while Pick two & place loses `11.8 pp`. On `valid_seen`, Pick two & place gains `16.7 pp`, while Look under light and Heat then place regress.

These results come from one training run and one evaluation seed. They demonstrate a working online Agentic RL loop and a positive result under these conditions. Stable conclusions require multiple training seeds, repeated evaluation and more checkpoint measurements.

The code below shows how environment interactions become inspectable JSONL records and figures.

## 7. How is the code organized?

Eight Python files make up the implementation. The pipeline is:

```text
发现并选择游戏
→ 为每个游戏创建 8 个独立环境
→ 并发采样完整工具轨迹
→ 计算 episode reward
→ 同游戏组内计算 advantage
→ 构造带 token mask 的 PyTRIO Datum
→ PPO backward + optimizer
→ 保存 checkpoint 与 SwanLab 指标
```

### 1. `data.py`: discover games and fix their order

[`data.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/data.py) discovers `game.tw-pddl` under `datasets/alfworld/json_2.1.1`, reads adjacent `traj_data.json`, filters task types and solvability, then builds `GameExample` records.

Games are shuffled with a fixed seed before training. `take_batch()` supports wrapping, so training continues when `--max-steps` exceeds one pass through the data.

Entry point: [`discover_games()`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/data.py#L80).

### 2. `protocol.py`: one tool and complete multi-turn context

[`protocol.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/protocol.py) defines `alfworld_step(action)`, the system prompt, tool-call parsing and observation formatting.

The key function is [`build_next_prompt()`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/protocol.py#L263). It appends assistant closing markers and tool observations directly to the prior actual prompt and returned completion tokens, without retokenizing historical assistant text.

This ensures:

1. The next prompt strictly extends the previous actual token sequence.
2. Rollout old log probabilities remain aligned with the assistant tokens used for training.

Tool-call parser: [`parse_assistant()`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/protocol.py#L100).

### 3. `environment.py`: K independent environments for one game

[`environment.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/environment.py) creates K independent TextWorld instances from one `game.tw-pddl`. `reset()` verifies identical initial observations and game files across branches so group comparisons really concern the same task.

`step()` receives K actions, advances the environment group and returns:

```text
observation
score
done / won
action 是否 admissible
下一步 admissible actions
```

This file also handles CPython 3.13+ compatibility for TextWorld 1.7.0's dynamic variable scope and safe shutdown of asynchronous workers. The project can consequently run under Python 3.14.

Core class: [`ALFWorldGroup`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/environment.py#L168).

### 4. `rollout.py`: advance long trajectories concurrently

[`rollout.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/rollout.py) implements the agent state machine.

The first turn's K trajectories share one prompt, so one request uses `num_samples=K`. Environment states then diverge. Each unfinished branch requests one candidate, with `sample_async` concurrency across trajectories and strict causal ordering within each trajectory.

Every step saves:

```text
prompt tokens
assistant completion tokens
rollout old logprobs
assistant text 与解析后的 action
tool observation
admissible / done / won
```

After all trajectories finish, `rollout_batch()` counts invalid actions, computes episode rewards and calls the advantage module.

Entry point: [`rollout_batch()`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/rollout.py#L469).

### 5. `advantages.py`: relative trajectory advantages within a game

[`advantages.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/advantages.py) groups trajectories by `group_id` and calculates `reward - group_mean`.

It also records degenerate groups with identical rewards. Their trajectories have zero advantage and skip ineffective backward computation.

Entry point: [`assign_advantages()`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/advantages.py#L18).

### 6. `train.py`: Datums, PPO and checkpoints

[`train.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/train.py) connects rollouts to PyTRIO.

[`build_trajectory_datum()`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/train.py#L249) applies one autoregressive shift to the complete multi-turn trajectory:

```text
model_input   = full_tokens[:-1]
target_tokens = full_tokens[1:]
```

It then constructs per-token old log probabilities and advantages:

```text
环境与工具 token: old_logprob = 0, advantage = 0
assistant token:  使用 rollout old_logprob 与 trajectory advantage
```

Each update in [`main()`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/train.py#L489) obtains the latest LoRA sampler, performs rollouts, filters zero-advantage trajectories, makes one PPO backward call and one optimizer step, then logs to SwanLab.

`save_checkpoint()` saves both state and sampler weights. Training trajectories remain in process memory and are not written separately to disk.

### 7. `eval.py`: concurrent evaluation with complete trajectory records

[`eval.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/eval.py) shares training's prompt, protocol, environments and rollout implementation.

Base evaluation leaves `model_path` empty; checkpoints use `trio://.../sampler_weights/...`. Evaluation runs `games-per-batch` games concurrently with one trajectory each, then separately aggregates `valid_seen` and `valid_unseen` metrics:

```text
success rate
reward mean
平均交互步数
truncated rate
工具格式正确率
admissible action rate
平均非法动作数
六类任务成功率
```

Unlike training, evaluation uses [`trajectory_record()`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/rollout.py#L585) to write every complete game to JSONL for later review.

### 8. `analysis.py`: plot results from JSONL

[`analysis.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/analysis.py) reads the final `type=summary` records from three evaluation JSONL files, checks model names and game counts, and generates the 1×2 checkpoint figure.

The left panel compares `valid_seen / valid_unseen` success. The right combines overall success with mean invalid actions.

The eight files can be summarized as follows:

| File | Responsibility |
| --- | --- |
| [`data.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/data.py) | Discover, filter, shuffle and cycle through games |
| [`protocol.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/protocol.py) | Tool protocol, full history, parsing and token-prefix extension |
| [`environment.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/environment.py) | K game branches, state transitions, compatibility and cleanup |
| [`rollout.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/rollout.py) | Asynchronous sampling, interaction, trajectory state, rewards and JSONL |
| [`advantages.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/advantages.py) | Same-game relative trajectory advantages |
| [`train.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/train.py) | Datums, token masks, PPO, optimizer, SwanLab and checkpoints |
| [`eval.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/eval.py) | Base/checkpoint evaluation, aggregated metrics and trajectory JSONL |
| [`analysis.py`](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/alfworld/analysis.py) | Read summaries and plot checkpoint comparisons |

Together these files connect game discovery, online interaction, relative credit assignment and independent evaluation.

## 8. Summary

At this point in the series, this experiment has the longest trajectories, most tool calls and most complex environment state.

Qwen3.5-4B samples 8 independent trajectories per ALFWorld game. Terminal success and invalid-action counts determine episode reward; same-game relative advantages and PPO update LoRA. The model must remember its actions through at most 50 interactions and 12K tokens of context, ending with a PDDL state that truly satisfies the task.

This provides a practical starting point for longer trajectories, more tools and more complex environments.
