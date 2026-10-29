# Reproducing Ten RL Algorithms, Chapter 9.2: TEMPO, Shorter Trajectory Segments and a Reasoning Critic

<div align="center">
  <img src="./images/封面.png" alt="An algorithm-level TEMPO reproduction" width="100%">
</div>

<div align="center">
  <a href="https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/experiments/tempo"><img alt="Code" src="https://img.shields.io/badge/Code-09--tempo-2563eb?style=flat"></a>
  <a href="https://arxiv.org/abs/2010.03768"><img alt="Dataset" src="https://img.shields.io/badge/Env-ALFWorld-f59e0b?style=flat"></a>
  <a href="https://studio.dots.ai/dots/tempo-blog.html"><img alt="TEMPO" src="https://img.shields.io/badge/TEMPO-Dots%20Blog-d94a45?style=flat"></a>
  <a href="https://docs.pytrio.com/docs"><img alt="PyTRIO" src="https://img.shields.io/badge/PyTRIO-0.2.8-7c3aed?style=flat"></a>
  <a href="https://www.zhihu.com/people/feng-qi-xia-pian" target="_blank"><img alt="Zhihu" src="https://img.shields.io/badge/Zhihu-知乎-4362f6?style=flat"></a>
  <a href="https://www.xiaohongshu.com/user/profile/63c2055e000000002502c58c" target="_blank"><img alt="Rednote" src="https://img.shields.io/badge/Rednote-小红书-e93c49?style=flat"></a>
  <a href="https://github.com/KMnO4-zx/agentic-rl-lab"><img alt="visitors" src="https://visitor-badge.laobi.icu/badge?page_id=KMnO4-zx.agentic-rl-lab.tempo"></a>
</div>

> **Code and reproduction resources**
>
> - Complete code: [KMnO4-zx/agentic-rl-lab/experiments/tempo](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/experiments/tempo)
> - Method source: [TEMPO: Test-Time-Scaled Value Estimation with Macro-Step Policy Optimization](https://studio.dots.ai/dots/tempo-blog.html), a technical blog from the Xiaohongshu Dots team
> - PyTRIO documentation: [docs.pytrio.com](https://docs.pytrio.com/docs)

This is chapter 9.2 in my series reproducing ten reinforcement learning algorithms.

A note on numbering: I originally reserved chapter 10 for our own Harness-RL. Until it is ready, chapter 9 continues as a 9.x series. Chapter 9.1 is [Vision GRPO](../vision-grpo/readme.md), and this chapter, 9.2, reproduces TEMPO.

This experiment began differently from the earlier chapters. **At the time of this note, TEMPO had no published paper**, only a Dots technical blog. The blog explained the mechanisms but omitted hyperparameters, training curves and ablations. I therefore aimed for an **algorithm-level reproduction**: implement the mechanisms that distinguish TEMPO from ordinary GRPO and verify them in a real environment, without pursuing leaderboard scores. I replaced the original ARC-AGI-3 environment with ALFWorld because ARC-AGI-3 used a paid API without exposed server-state restoration. The interaction infrastructure comes from chapter 8.

First, what problem is TEMPO trying to solve?

## 0. Two obstacles in long-horizon tasks

The earlier Agentic RL experiments share an implicit assumption: trajectories are short enough. Search-R1 retrieves information for a few rounds, ReTool executes several code snippets, and ALFWorld resolves within fifty steps. When an agent takes on tasks lasting hours or days, the GRPO approach encounters two obstacles.

**The first is delayed feedback.** Outcome rewards are available only when a trajectory ends. If a rollout takes eight hours, generating that training signal takes at least eight hours. Much of the computational effort is spent waiting for the outcome.

**The second is credit assignment.** In a trajectory hundreds of turns long, the third action is separated from success or failure by hundreds of steps. Subtracting a group mean provides little guidance about which early decision helped.

The textbook PPO-style solution is a value head that predicts future returns at intermediate states, allowing unfinished trajectories to provide training signals. A scalar value head, however, uses **one forward pass with a fixed computational budget**. In long-horizon tasks, valuing a state can itself require reasoning: are the actor's assumptions correct, is its search direction viable, and which obstacles remain? A small scalar head has limited capacity to perform that analysis.

TEMPO's answer is in its name: Test-Time-Scaled Value Estimation with Macro-Step Policy Optimization. It combines three main designs.

## 1. TEMPO's three main designs

### 1.1 Macro-steps: optimize a segment instead of an entire episode

TEMPO defines **one macro-step** as H consecutive rounds of model–environment interaction. This replaces the full trajectory as the basic unit of rollout and optimization. For each update, the actor resumes from a saved intermediate state, the environment restores that state, and execution stops after H rounds.

![Comparing GRPO, PPO and TEMPO rollouts](./images/tempo-1.png)

*In this comparison, GRPO executes all T interaction rounds before receiving a terminal return. PPO also rolls out to T and estimates V(s) along the way with a scalar value head. TEMPO executes an H-round macro-step, calls a generative critic at the boundary, and saves selected endpoints as future starting states. Image source: [TEMPO blog](https://studio.dots.ai/dots/tempo-blog.html).*

The blog appendix provides an easily overlooked guarantee: after correcting distribution differences in historical prefixes, **the expected gradient from optimizing the current macro-step is equivalent to the policy gradient of the full trajectory**. With a fixed segmentation rule, the full gradient is a sum over M macro-steps. Uniformly sampling one segment and multiplying its contribution by M preserves that expectation. This requires both the prefix and segment to follow the current policy; stale prefixes require importance-sampling correction. Our first version omits that correction, as discussed below.

### 1.2 A generative critic: value estimation requires reasoning

TEMPO replaces a scalar value head with a generation task. The critic reads the current state and complete interaction history, analyzes what the actor has learned about the environment, its attempted solution and unresolved obstacles, then outputs a numerical value. Harder states can receive longer reasoning and further reflection. **Test-time scaling can benefit the critic as well as the actor.**

The blog illustrates this with a knight-placement task:

![Rules of the knight-placement task](./images/tempo-2.png)

*The actor must infer hidden constraints through interaction. Starting from two preplaced pieces, it must place six more so that no pair can attack each other with a knight's move. Image source: [TEMPO blog](https://studio.dots.ai/dots/tempo-blog.html).*

Two 64-round trajectories start from the same state and receive the same environment reward, 0, but reach very different situations:

![Equal environment rewards but different trajectory values](./images/tempo-4.png)

*Branch A mistakenly treats the presence of attacks as the goal. All its candidate layouts require an attack, excluding valid solutions from its search space. Comparing the trajectory with the game source, the critic assigns V̂=2.6. Branch B revisits the history, rejects that assumption, discovers that knights on same-colored squares do not attack each other, and proposes a viable solution. The critic assigns V̂=4.6. The value difference is 2.0, while the environment-reward difference is 0. Image source: [TEMPO blog](https://studio.dots.ai/dots/tempo-blog.html).*

This example also explains **privileged information**: the critic can access internal environment states, hidden rules or game source code. The actor cannot see this information; the critic uses it to verify its evaluation. Its judgment of branch A comes from checking the source.

Why estimate values only at macro-step boundaries? A generative critic produces an entire reasoning sequence and costs much more than a scalar value-head forward pass. Running it after every token would be expensive. One call every H rounds trades feedback granularity against computation.

### 1.3 The actor is also its own critic: two roles, one GRPO framework

Once value estimation becomes generation, TEMPO can maintain **one set of parameters**. The model acts during interaction, then switches to a critic prompt at the macro-step boundary. Prompts, context and rewards distinguish the roles, while both share a sampler and weight update.

Both roles consequently fit into the same GRPO framework. Their losses have the same form; the reward source differs. Actor learning uses environment-derived rewards, while critic learning uses estimation error. In PyTRIO, both can enter one `forward_backward` batch and share a LoRA adapter.

## 2. Training signals: four formulas and a warm-up

The mathematical core consists of four signals. From one starting state, sample N branches of H rounds each:

```text
① 分支 return       Rₙ = rₙ(段内环境奖励) + V̂(终点状态)      # 截断的尾部由 critic 补齐
② TD target         G   = mean(R₁ .. Rₙ)                    # "这个起点值多少"
③ actor advantage   Aₙ  = Rₙ − mean(R)                       # 标准 GRPO 组内减均值
④ critic reward     rₖ  = −|V̂ₖ − G| / R_max                 # 同一状态 K 次独立估值，谁准谁得分
```

Signal ③ updates the actor's H rounds of behavior. Signal ④ is centered within the group to update the critic's reasoning. Signal ② is the temporal-difference (TD) target: observed rewards within a segment plus a bootstrap estimate. Error is normalized by the remaining-return range R_max, keeping signals comparable across states.

![Two critic value targets and their training procedures](./images/tempo-3.png)

*Top: TD targets combine rewards from several macro-step branches with endpoint values. Bottom: warm-up targets use Monte Carlo returns from complete offline trajectories. Given G, both methods assign rewards from estimation error and update the critic with GRPO. Image source: [TEMPO blog](https://studio.dots.ai/dots/tempo-blog.html).*

Starting directly with TD is risky: an untrained critic's inaccurate bootstrap targets can propagate errors toward earlier states. TEMPO first performs **value warm-up**. Complete offline trajectories provide actual outcomes, or Monte Carlo returns, as G. The critic learns to assess states before entering TD. Our training loop therefore has two phases: critic-only warm-up, followed by joint actor–critic TD updates that store and reuse endpoint states.

## 3. Our reproduction strategy: verify the algorithm

At the time of this experiment, the paper was unavailable and the blog withheld H, N, K and experimental numbers. Reproducing benchmark scores was therefore outside scope. My goal was to **implement each mechanism distinguishing TEMPO from ordinary GRPO and run it in a real environment**. The choices are:

| Original TEMPO setting | This implementation | Reason |
|---|---|---|
| ARC-AGI-3, 25 games | ALFWorld train split, 3553 games | ARC-AGI-3 uses a paid API without exposed server-state restoration; ALFWorld is local, free and supported by chapter 8's infrastructure. |
| Very long horizons, hours or days | T=50 rounds, H=10, M=5 segments per episode | Enough for mechanism checks, with manageable context length. |
| Generative critic with privileged information | One model with a critic prompt; the `game.tw-pddl` walkthrough supplies privileged information | Corresponds to the critic seeing game source in the original method. |
| Critic information hidden from the actor | Actor prompts contain only the task and interaction history | |
| TD targets, error rewards and GRPO | Implemented with N=4 branches, K=4 value samples and R_max=1 | Binary success rewards keep the remaining-return range at 1. |
| Value warm-up | Train the critic first with complete-trajectory Monte Carlo returns | |
| Importance correction for prefix distribution shift | Omitted in the first version | The appendix's equivalence holds for on-policy prefixes. Immediate reuse may limit staleness, but does not remove this approximation. |
| Reward | Binary `won` signal | We omit chapter 8's invalid-action penalty to retain R_max=1. |

A useful detail is that ALFWorld's `game.tw-pddl` files contain a `walkthrough` field: an expert action sequence for each game. It supplies privileged information the actor cannot see but the critic can read.

## 4. The main engineering problem: saving and restoring an environment

Macro-step training requires **saving a complete intermediate state and restoring it after subsequent training updates**. This is a new layer in the repository; chapter 8's `EnvironmentState` only captured the initial snapshot after reset.

Each saved state needs two snapshots:

```text
环境侧：action_history   —— 按顺序重放即可恢复游戏内部状态
token 侧：token_prefix   —— 续跑用的真实 token 前缀，保证下一轮 prompt
                            仍然是历史真实 token 序列的严格前缀扩展
外加：messages（chat 历史）、round_index、environment_seed、当前 observation
```

Restoration relies on TextWorld determinism: the same game file and action sequence reach the same state. To restore, create fresh environments, reset, and replay each historical action. Then compare the observation **character for character with the saved observation**. A mismatch raises an error and refuses to train on the state.

We tested this separately in real environments. For 3 games, we executed 5–6 walkthrough steps and replayed them in fresh environments; observations matched exactly. Taking one further action after restoration also produced identical behavior in two independently restored environments. This was the part I most wanted to verify before trusting the pipeline.

At each macro-step boundary, branch endpoints and their histories enter a `StateStore`, a FIFO store with capacity 256. Later training samples uniformly from it. A game can then advance across several updates while each rollout pays for only 10 interaction rounds.

## 5. Running training

Data downloads into this chapter's own directory, independently of chapter 8:

```bash
uv run --extra alfworld alfworld-download --data-dir "$PWD/datasets/alfworld"
```

A small smoke run checks the pipeline; actual service cost depends on usage:

```bash
uv run --extra alfworld python train.py \
    --warmup-updates 4 --td-updates 4 \
    --warmup-games-per-batch 4 --states-per-batch 4 \
    --branches 4 --critic-samples 4 --endpoint-samples 1 \
    --swanlab-mode online
```

For the full mechanism-check configuration, increase `--td-updates` to 40. The parameters are:

| Parameter | Default | Meaning |
|---|---|---|
| `--macro-rounds` | 10 | H: rounds per macro-step |
| `--branches` | 4 | N: branches per starting state, the actor group size |
| `--critic-samples` | 4 | K: value samples per state, the critic group size |
| `--states-per-batch` | 4 | Starting states per TD update; Datum count = states × (N + K) |
| `--warmup-updates` / `--td-updates` | 4 / 40 | Updates in each phase; their sum is the total progress |
| `--endpoint-samples` | 2 | Value samples averaged at each endpoint to form returns; not a training group |

## 6. Observations from the run

The first warm-up update already showed the complete pipeline running:

```text
warmup=1 critic/abs_error_mean=0.606 warmup/boundary_states=22
         warmup/steps_mean=19.688 warmup/success_rate=0.750 loss=0.0513
```

- The 16 trajectories ended after 19.7 rounds on average and yielded 22 boundary states. An average trajectory shorter than two segments is expected here.
- `abs_error_mean=0.606`: the untrained critic was not yet predicting success well. Warm-up aims to reduce this error.
- `success_rate=0.75`: with available actions included in the prompt, the 4B model started stronger than I expected.

We track four indicators when checking the mechanism:

| Metric | Desired observation |
|---|---|
| `critic/abs_error_mean` | Declines during warm-up |
| `critic/value_corr` | Correlation between V̂ and actual success increases |
| `actor/value_gap_zero_reward` | The critic distinguishes values of branches with equal segment rewards, analogous to knight placement |
| `td/fresh_states` decreases from the second TD update | Stored states are being restored and reused, allowing nonterminal segments to supply gradients |

Actor observations from a smoke run of 8 updates, 4 warm-up and 4 TD:

![Actor metrics from the SwanLab smoke run](./images/swanlab-td.png)

*The first 4 steps are warm-up and the next 4 are TD. `actor/segment_reward_mean` rises from 0.62 to 0.85. `actor/degenerate_group_rate` falls from 0.19 to zero: initially, roughly a fifth of starting-state groups had identical returns and no relative training signal. By the fourth TD step, none did. This is consistent with V̂ separating branch returns and supplying a usable actor signal.*

![PyTRIO usage for this smoke run](./images/pytrio-consume.png)

*This run cost ¥15.02, compared with the four-digit cost of chapter 8's full training run. The smaller algorithm-level reproduction was substantially cheaper.*

The critic curves are central to checking this implementation:

![Critic metrics from the SwanLab smoke run](./images/swanlab-critic.png)

*The five panels show `critic/abs_error_mean`, `critic/value_target_mean`, `critic/value_target_std`, `critic/value_corr` and `critic/parse_failure_rate`. Estimation error falls from 0.674 to 0.315, decreasing by roughly two fifths during three warm-up steps and continuing downward in TD. Warm-up value correlation rises from 0.60 to 0.72. Parsing failures decline from 7.3% to around 1%, indicating more reliable `<value>` formatting.*

Alongside these critic changes, the actor's `degenerate_group_rate` approaches zero. In this run, improving value estimates coincided with more differentiated returns across actor branches, with both roles sharing one parameter set.

## 7. Code organization

| File | Responsibility |
|---|---|
| `protocol.py` | ALFWorld's single-tool protocol and multi-turn token prefix, reused from chapter 8 |
| `environment.py` | Multiple environment branches for one game, reused from chapter 8 |
| `data.py` | Game discovery and `load_walkthrough` for privileged information |
| `states.py` | **New:** MacroState's two snapshots, StateStore, and boundary/endpoint extraction |
| `critic.py` | **New:** Critic prompts, K value samples, strict `<value>` parsing and fallback |
| `rollout.py` | **New:** Replay restoration and concurrent N-branch macro-step rollouts |
| `tempo.py` | **New:** Assemble Rₙ, G, error rewards, advantages and training Datums |
| `train.py` | **New:** The warm-up → TD training loop |

Follow this TD-update data flow while reading `train.py`:

```text
① 取起点   StateStore 均匀抽样，不足用新游戏开局补
② 恢复环境 新开环境 → reset → 重放 action_history → observation 逐字校验
③ 跑一段   N 分支并发走 H 轮，停在 macro_boundary
④ 估终点   每个非终局分支的终点 → critic prompt → V̂（终局分支恒为 0）
⑤ 拼信号   Rₙ = rₙ + V̂ₙ → G = mean(R) → 双方各自组内中心化
⑥ 更新     actor + critic 的 Datum 合并 forward_backward（importance_sampling）
⑦ 入库     边界终点连同历史写回 StateStore，供后续轮次续跑
```

The server uses no custom loss. TEMPO's changes—assembling returns and G, calculating error rewards and masking—happen locally before constructing `Datum.loss_fn_inputs`. This lets actor and critic use the same GRPO loss.

An actor branch's Datum is a complete sequence containing the saved prefix and all turns within the segment. Environment-observation tokens are context only, with zero logprob/advantage entries; only assistant-generated tokens inside the segment carry the branch advantage. A critic Datum works similarly: the value prompt is context, while the generated reasoning and `<value>` output carry its advantage.

## 8. Boundaries and next steps

This reproduction leaves three items unresolved:

1. **Prefix importance-sampling correction.** Starting prefixes may come from older actor versions. Strict equivalence requires weighting advantages by prefix probability ratios. Immediate endpoint reuse may reduce staleness, but does not guarantee negligible bias. A possible implementation is to store the sum of old assistant-token log probabilities with each state, call `compute_logprobs` for current values at each starting state, and multiply the ratio into advantages.
2. **An ARC-AGI-3 comparison.** Once a paper and hyperparameters are available, the environment layer can be replaced for a closer reproduction.
3. **Longer horizons.** T=50 checks the mechanism. Extending T beyond 100 brings context length into focus as a practical constraint.

## Summary

TEMPO combines macro-step optimization, a reasoning-based generative critic for the truncated future, and shared actor–critic parameters trained with GRPO. The most striking implementation detail is how much happens in local orchestration: saving states, restoring them and assembling signals. The service still needs standard sample, logprob and forward_backward operations. This makes the mechanisms practical to inspect and implement in the training loop.

This experiment cost **¥15.02**, far below chapter 8's four-digit full-training cost. Once the paper and hyperparameters are public, I expect to revisit it in the 9.x series.
