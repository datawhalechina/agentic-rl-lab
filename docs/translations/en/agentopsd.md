# Reproducing Ten RL Algorithms, Chapter 9.3: AgentOPSD, an Agent as Its Own Teacher

> **Quick start:** See [start.md](./start.md) for dependencies, a smoke run, full training, checkpoint evaluation and result plotting.

<div align="center">
  <img src="./images/封面.png" alt="AgentOPSD: an agent as its own teacher" width="100%">
</div>

<div align="center">
  <a href="https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/experiments/agentopsd"><img alt="Code" src="https://img.shields.io/badge/Code-09--AgentOPSD-2563eb?style=flat"></a>
  <a href="https://arxiv.org/abs/2608.05987"><img alt="AgentOPSD" src="https://img.shields.io/badge/Paper-AgentOPSD-d94a45?style=flat"></a>
  <a href="https://arxiv.org/abs/2010.03768"><img alt="ALFWorld" src="https://img.shields.io/badge/Env-ALFWorld-f59e0b?style=flat"></a>
  <a href="https://docs.pytrio.com/docs"><img alt="PyTRIO" src="https://img.shields.io/badge/PyTRIO-0.2.8-7c3aed?style=flat"></a>
  <a href="https://www.zhihu.com/people/feng-qi-xia-pian" target="_blank"><img alt="Zhihu" src="https://img.shields.io/badge/Zhihu-知乎-4362f6?style=flat"></a>
  <a href="https://www.xiaohongshu.com/user/profile/63c2055e000000002502c58c" target="_blank"><img alt="Rednote" src="https://img.shields.io/badge/Rednote-小红书-e93c49?style=flat"></a>
  <a href="https://github.com/KMnO4-zx/agentic-rl-lab"><img alt="visitors" src="https://visitor-badge.laobi.icu/badge?page_id=KMnO4-zx.agentic-rl-lab.agentopsd"></a>
</div>

> **Code and reproduction resources**
>
> - Complete code: [KMnO4-zx/agentic-rl-lab/experiments/agentopsd](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/experiments/agentopsd)
> - Paper: [Fine-Grained Agentic Reinforcement Learning via On-Policy Self-Distillation](https://arxiv.org/abs/2608.05987)
> - Upstream implementation: [ZethWang/AgentOPSD](https://github.com/ZethWang/AgentOPSD/tree/0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b)
> - ALFWorld: [Paper](https://arxiv.org/abs/2010.03768) / [Official repository](https://github.com/alfworld/alfworld)
> - PyTRIO documentation: [docs.pytrio.com](https://docs.pytrio.com/docs)

This is chapter 9.3 in my series reproducing ten reinforcement learning algorithms. Chapter 9 has become a 9.x series: [9.1 Vision GRPO](../vision-grpo/readme.md) solves geometry problems from images, and [9.2 TEMPO](../tempo/readme.md) divides long trajectories into macro-steps. This chapter addresses credit assignment: when an agent succeeds after dozens of actions, which actions mattered? In AgentOPSD, the current agent first performs a real rollout without skills. The same parameter snapshot then reviews every turn with skills available. No additional critic is trained, and the teacher does not rewrite a reference trajectory. Final reward determines the trajectory's learning direction; the self-teacher identifies locally important actions. We completed 80 training steps with Qwen3.5-4B, PyTRIO and text-only ALFWorld.

I also see this as a possible early component of recursive self-improvement (RSI). Imagine a privately deployed model working through a harness such as Codex or Claude Code during the day and retaining its trajectories. At night, successes, failures and user corrections could be summarized into reusable skills. The same model could use those skills to reassess its original trajectories, assign turn credit with AgentOPSD, and update weights through infrastructure such as PyTRIO or Tinker. After evaluation without skills, the new model could return to work. In that proposed workflow, skills expose experience to the model, while training aims to incorporate it into the weights.

*This remains some distance from complete RSI. My view is that it offers an elegant, practical path to investigate while more mature continual-learning methods develop.*

![Work by day, learn at night: a proposed continual-learning loop](./images/rsi.png)

*The proposed loop generates real trajectories through a harness, summarizes skills, trains with AgentOPSD and evaluates without skills before using the next model.*

## 0. Results first: overall success from 58.8% to 65.7%

Base, Step 40 and Step 80 use exactly the same 274 games: 140 `valid_seen` and 134 `valid_unseen`. No skills enter evaluation prompts. `analysis.py` also checks every `(split, game_id)` across the three JSONL files so that checkpoint comparisons use identical tasks.

![ALFWorld evaluation of AgentOPSD Base, Step 40 and Step 80](./images/agentopsd_checkpoint_evaluation.png)

| Model / checkpoint | Valid Seen | Valid Unseen | Overall success | Mean invalid actions |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B Base | 60.0% | 57.5% | 58.8% | 7.07 |
| AgentOPSD Step 40 | 61.4% | 62.7% | 62.0% | 6.74 |
| **AgentOPSD Step 80** | **64.3%** | **67.2%** | **65.7%** | **5.68** |

Step 80 compared with Base:

- Overall success improves by `6.9` percentage points, from `161 / 274` to `180 / 274` successful games.
- `valid_unseen` improves by `9.7` percentage points, more than the seen split.
- Mean invalid actions per game fall from `7.07` to `5.68`, a reduction of `1.39`.
- Step 40 already improves on Base, with further progress at Step 80.

## 1. What is AgentOPSD?

Ordinary GRPO operates at trajectory level.

It samples a group of rollouts for the same task, assigning positive advantages to successful trajectories and negative advantages to failed ones when both outcomes occur. All action tokens within a trajectory typically inherit the same sequence-level signal.

In ALFWorld, an agent might open two wrong cabinets, wander for a dozen steps, then find an apple and put it in the refrigerator. The successful terminal outcome encourages the earlier detours alongside the decisive final actions. This coarse credit assignment becomes more pronounced in longer trajectories.

AgentOPSD adds local OPSD judgments to GRPO's trajectory-level direction:

![AgentOPSD's agent loop, turn-level OPSD and sequence-level GRPO](./images/AgentOPSD-paper.png)

*The skill-conditioned self-teacher in the middle computes token gaps on each executed student turn, then accumulates them into turn credit. GRPO on the right still supplies the trajectory's final direction. Source: [AgentOPSD paper](https://arxiv.org/abs/2608.05987).*

The loop has three roles. Both student and self-teacher use **the current model**:

- **Student:** interacts with the environment without skills, matching deployment conditions.
- **Self-teacher:** uses the same current parameter snapshot but additionally sees task skills from the SkillBank.
- **Verifier:** checks terminal environment success and assigns a binary trajectory outcome.

The teacher **does not generate a better replacement trajectory**. It recomputes probabilities along the exact trajectory and action tokens produced by the student. The log-probability difference for a token under the two contexts supplies local evidence about whether knowing the skill would make that action more likely.

This also differs from the [ordinary OPSD in chapter 4](../opsd/readme.md):

| Method | Main supervision | Signal granularity | Teacher context |
| --- | --- | --- | --- |
| GRPO | Terminal rewards across trajectories for the same task | Complete trajectory | No teacher |
| OPSD | Same-model rescoring of student completions | Reasoning tokens | Reference solution |
| AgentOPSD | Terminal GRPO signal plus local self-teacher evidence | Agent turn | Task-specific skills |

A concise description is:

> **Final reward sets the learning direction. The self-teacher decides which turns within a long trajectory receive slightly more or less emphasis.**

## 2. The complete training process

The following Chinese diagram summarizes the mechanism used by this implementation.

![Simplified AgentOPSD mechanism and turn-level credit](<./images/AgentOPSD · Turn-Level Credit.png>)

### 2.1 The student acts without skills

Each training step freezes the current parameter snapshot. For one ALFWorld game, create 8 independent TextWorld environments and sample 8 trajectories from the same task and initial state.

The student prompt contains only the task, historical observations, currently available actions and the `alfworld_step` protocol. It never sees the SkillBank, keeping training rollouts aligned with the skill-free evaluation setting.

Each trajectory has at most 50 turns. The environment supplies a binary success outcome, and training uses a pure 0/1 reward without an additional invalid-action penalty.

### 2.2 The same model reviews the same trajectory with skills

After student rollout, no optimizer step has occurred. The code reuses the frozen sampling snapshot, replacing only the initial prompt with a teacher version that includes:

- A general skill shared across all ALFWorld tasks.
- A task-type skill, such as cleaning, heating, cooling or pick-and-place.

The teacher makes one `compute_logprobs` call for the complete student trajectory. It changes no actions and does not rerun the environment; it rescores the student's actual assistant action tokens.

The implementation maps ALFWorld `task_type` deterministically to fixed skill files, without semantic retrieval. This is easier to audit and avoids exposing rewards or future observations to the teacher.

### 2.3 Token gaps become turn evidence

For every action token, compare log probabilities under the skill-conditioned teacher and skill-free student contexts. Tokens favored more by the teacher have positive gaps, and those favored less have negative gaps. Sum the token gaps within a turn to obtain its evidence.

Actual action spans must be retained. System/user content, environment observations and chat-template tokens are context; they must not enter teacher evidence or the policy gradient.

### 2.4 Accumulate evidence over time to identify important turns

AgentOPSD updates a relative belief in trajectory success in chronological order. Each turn's evidence affects that shared belief. The change caused by a turn, `ΔB` in the diagram, represents its local importance.

Earlier evidence decays with `gamma=0.95`. History still affects later decisions, without letting the first action dominate indefinitely.

This belief is relative support, not a calibrated success probability. Training uses changes between adjacent turns; the absolute value does not determine turn importance.

### 2.5 Terminal outcomes set direction; teacher adjustments stay bounded

The 8 trajectories for a game first receive standard GRPO sequence advantages from terminal binary rewards. Successful trajectories have positive direction and failed trajectories negative direction when outcomes vary. If all succeed or all fail, every advantage is 0 and the group supplies no policy gradient.

`ΔB` then adjusts each turn's strength. Turn weights are bounded to `[0.8, 1.2]`, and `reshape_lambda=0.5`, so final advantages remain between `0.9` and `1.1` times the original GRPO advantage.

The self-teacher can therefore emphasize or de-emphasize a turn by at most 10%. It cannot turn a successful trajectory's advantage negative or override the terminal environment outcome.

### 2.6 Update only assistant action tokens

Each complete multi-turn trajectory becomes a PyTRIO PPO Datum. Assistant action tokens receive their turn's reshaped advantage. Environment observations and system/user tokens receive zero advantage.

PyTRIO runs Qwen3.5-4B LoRA forward/backward passes and optimizer updates. Local code handles ALFWorld, student rollouts, skill selection, teacher alignment, turn credit and batch boundaries. Only after one PPO update does the next parameter snapshot begin.

## 3. How did we reproduce it?

We first identified the method's core invariants, then connected them to chapter 8's verified ALFWorld long-trajectory infrastructure instead of translating the upstream trainer line by line.

### 3.1 Reuse a reliable environment layer

From [experiments/alfworld](../alfworld/readme.md), we reuse independent branches for one game, strictly extending actual-token prefixes, and the causal sequence action → environment step → observation → next action.

These details determine whether teacher log probabilities really align. Rerendering history or silently truncating context would change the trajectory the teacher reviews relative to what the student saw.

### 3.2 Pin the SkillBank and task mapping

ALFWorld skill assets come from a fixed upstream AgentOPSD commit. We retain the general skill and five task-specific skills, with an explicit `task_type → skill` mapping for six ALFWorld task types.

Skills enter teacher prompts only during training. Student rollouts, Base evaluation and checkpoint evaluations all remain skill-free, distinguishing learning in the weights from access to extra guidance.

### 3.3 Save alignment evidence during rollout

Each turn retains action tokens, start/end spans, student old log probabilities and the policy snapshot ID. After adding skills to the initial prefix, the teacher reconstructs action spans with relative offsets and checks token counts and positions for exact agreement.

Because each ALFWorld turn strictly extends the actual prior token sequence, one full teacher request per trajectory suffices; dozens of separate per-turn requests are unnecessary.

### 3.4 Check the core mechanism with built-in PPO

AgentOPSD's main change happens before the loss: local code converts teacher gaps into per-turn advantages, then passes them to PyTRIO's built-in PPO. Ratio clipping uses `[0.8, 1.24]`, with one PPO epoch per rollout to keep updates close to the on-policy setting.

We do not exactly reproduce full-vocabulary entropy, full-parameter training or some paper-specific reduction details. This is an **algorithm-level reproduction**, focusing on the loop of same-snapshot self-teaching, turn credit, action-only masks and skill-free evaluation.

The full experiment uses:

| Item | Configuration |
| --- | --- |
| Base Model | `Qwen/Qwen3.5-4B` |
| Training duration | 80 updates |
| Tasks per update | 16 ALFWorld games |
| Trajectories per task | 8, totaling 128 per update |
| Maximum interaction length | 50 turns |
| Student trajectory budget | 14,336 tokens, plus 2,048 reserved for the teacher skill prefix |
| LoRA rank / learning rate | 32 / `4e-6` |
| Turn credit | `gamma=0.95`，`lambda=0.5`，`weight_bound=0.2` |
| PPO clip | `[0.8, 1.24]` |
| Checkpoint | Step 40、Step 80 |
| Fixed evaluation | 140 seen + 134 unseen; temperature `0.01`, seed `42`, no skills |

## 4. Quick start

Run all commands from the repository root. Python `>=3.13` is required. Data, environments and orchestration run locally; PyTRIO performs remote sampling and LoRA training.

### 4.1 Install dependencies and download ALFWorld

```bash
git clone https://github.com/KMnO4-zx/agentic-rl-lab.git
cd agentic-rl-lab

uv sync --extra alfworld
trio login
swanlab login

uv run --extra alfworld alfworld-download \
    --data-dir "$PWD/experiments/agentopsd/datasets/alfworld"
```

### 4.2 Begin with a minimal smoke run

This small command uses paid remote calls to verify the environment, student rollout, teacher scoring, PPO and checkpoint pipeline:

```bash
uv run --extra alfworld python experiments/agentopsd/train.py \
    --max-steps 1 \
    --tasks-per-update 1 \
    --group-size 4 \
    --max-turns 20 \
    --save-every 1 \
    --seed 3 \
    --swanlab-mode disabled
```

All 4 trajectories for one game may succeed or fail together. Then sequence advantages are zero; the script records rollout metrics but skips backward. Increase `--tasks-per-update` when checking the complete parameter-update path.

### 4.3 Run the full 80-step experiment

```bash
uv run --extra alfworld python experiments/agentopsd/train.py \
    --base-model Qwen/Qwen3.5-4B \
    --max-steps 80 \
    --tasks-per-update 16 \
    --group-size 8 \
    --max-turns 50 \
    --max-trajectory-tokens 14336 \
    --max-action-tokens 512 \
    --temperature 1.0 \
    --top-p 1.0 \
    --teacher-concurrency 16 \
    --gamma 0.95 \
    --reshape-lambda 0.5 \
    --weight-bound 0.2 \
    --ppo-clip-low 0.8 \
    --ppo-clip-high 1.24 \
    --lora-rank 32 \
    --learning-rate 4e-6 \
    --save-every 40 \
    --swanlab-mode online
```

At Steps 40 and 80 and at training completion, two paths are saved:

- `state`: includes optimizer state for resuming training.
- `sampler weights`: for sampling and evaluation with `eval.py`.

### 4.4 Evaluate the base model

```bash
uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/base-qwen35-4b-eval.jsonl \
    --swanlab-mode disabled
```

### 4.5 Evaluate Steps 40 and 80

These commands use sampler weights from the documented experiment. For your own run, replace them with the path printed after `Saved sampler weights:`.

```bash
uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path 'trio://run_sschmbprfwg0/sampler_weights/agentopsd-alfworld-qwen35-4b-step-40-weights' \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/checkpoint-eval-step40.jsonl \
    --swanlab-mode disabled

uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path 'trio://run_sschmbprfwg0/sampler_weights/agentopsd-alfworld-qwen35-4b-step-80-weights' \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/checkpoint-eval-step80.jsonl \
    --swanlab-mode disabled
```

Each JSONL contains 274 complete trajectories and a final unified summary. Preserve existing outputs by choosing a new filename, or deliberately pass `--overwrite-output` if you intend to replace them.

### 4.6 Generate the checkpoint comparison plot

```bash
uv run python experiments/agentopsd/analysis.py
```

The script reads Base, Step 40 and Step 80 summaries, verifies identical game IDs, then generates:

```text
experiments/agentopsd/images/agentopsd_checkpoint_evaluation.png
```

## 5. What happened during training?

Fixed evaluation measures checkpoint performance. SwanLab curves check whether AgentOPSD's intermediate signals actually enter training.

### 5.1 Online reward is noisy and cannot replace fixed evaluation

![Reward metrics during 80 steps of AgentOPSD](./images/swanlab-reward.png)

`reward/mean` and `reward/success_rate` fluctuate throughout 80 steps without a textbook monotonic rise. Each update samples different ALFWorld games with different difficulties. A batch can also contain all-success or all-failure groups with no group-relative gradient.

These curves help check rollout health and the presence of both successes and failures. Capability changes require the fixed 274-game evaluation reported above.

### 5.2 Turn-level credit is active

![AgentOPSD evidence, belief changes and turn weights](./images/swanlab-opsd.png)

These diagnostics are specific to AgentOPSD:

- `evidence_abs_mean` stays nonzero: skill-conditioned teacher and skill-free student probabilities differ for the same actions.
- `delta_b_abs_mean` stays nonzero: accumulating that evidence over time produces turn-level belief changes.
- `weight_mean` stays near 1.0: reshaping does not uniformly raise or lower all trajectory weights.
- `weight_min` repeatedly reaches the 0.8 lower bound: some turns are actively downweighted, producing distinct final advantages across turns.

These metrics establish that the credit-assignment pipeline runs, but they are training diagnostics rather than capability measures. Large evidence or `ΔB` does not imply better gameplay. The outcome evidence remains skill-free checkpoint evaluation.

### 5.3 Long prefixes and teacher rescoring drive the cost

![PyTRIO usage for the full 80-step AgentOPSD run](./images/pytrio-consume.png)

The training session records:

| Item | Usage |
| --- | ---: |
| Prefilling | 762.21M tokens, including 549.26M cached |
| Train | 30.52M tokens |
| Sample | 9.98M tokens |
| **Cost** | **¥675.69** |

The `762.21M` prefill count stands out. Every ALFWorld turn carries an increasingly long history, and the self-teacher then rescores the complete trajectory with a skill-conditioned context. This usage pattern matches the long-trajectory plus same-trajectory teacher-scoring design.

Cache hits cover 549.26M tokens and save substantial repeated-prefix work, yet this run still costs much more than earlier short-answer tasks. Agentic RL can spend heavily rereading the history needed to reach each action; the action tokens entering the loss are a smaller part of total usage.

## 6. How should these results be interpreted?

I see three levels of evidence with different strengths.

The strongest is **an operational engineering loop**: one snapshot performs skill-free rollouts and skill-conditioned reviews, action spans align, turn credit reaches action-only PPO, and saved checkpoints support independent skill-free evaluation.

The second is the positive result of this one experiment. Both Step 40 and Step 80 exceed Base on fixed evaluation. Step 80 completes 19 additional games with fewer invalid actions, and gains more on the unseen split. The improvement is present in the saved skill-free policy, beyond access to teacher skills during training.

The third requires controls: **how much comes from AgentOPSD's turn credit, and how much from PPO/GRPO training itself at the same budget?** The next experiment should add a `reshape_lambda=0` GRPO control with matched game order, seed, rollout count and token budget, then evaluate Base, GRPO and AgentOPSD across multiple seeds.

There are also clear differences from the paper: its models and full-parameter training stack differ, while this version uses Qwen3.5-4B LoRA and built-in PyTRIO PPO. Full-vocabulary entropy and other paper-specific details are not replicated exactly. This remains an **algorithm-level AgentOPSD reproduction on PyTRIO + ALFWorld**, rather than a reproduction of the paper's benchmark scores.

## 7. Summary

My one-sentence description of AgentOPSD is:

> **A skill-informed version of the same model reviews the path it just took and identifies the turns that mattered.**

GRPO retains the final direction, encouraging successful trajectories and discouraging failed ones. The self-teacher makes bounded adjustments to individual turns within that direction, retaining the environment verifier's authority while giving sparse terminal rewards finer credit allocation.

We completed the loop in 80 steps and 10,240 training rollouts. On the fixed 274-game evaluation, overall success rises from 58.8% to 65.7%, and mean invalid actions fall from 7.07 to 5.68. The result is positive and the cost substantial. It provides a working, inspectable starting point for matched GRPO controls and multi-seed experiments.

## 8. What does each code file do?

| File | Responsibility |
| --- | --- |
| [`data.py`](./data.py) | Discover games and fix splits, task types and sampling order |
| [`protocol.py`](./protocol.py) | Define `alfworld_step`, multi-turn messages and strict token prefixes |
| [`environment.py`](./environment.py) | Create isolated TextWorld branches for one game |
| [`skills.py`](./skills.py) | Load the pinned SkillBank and map `task_type → skill` |
| [`rollout.py`](./rollout.py) | Run concurrent student trajectories, retaining action spans and old log probabilities |
| [`teacher.py`](./teacher.py) | Add skills, rescore complete trajectories and recover per-turn scores |
| [`advantages.py`](./advantages.py) | Compute GRPO sequence advantages and AgentOPSD turn credit |
| [`loss.py`](./loss.py) | Build PPO Datums that train only assistant action tokens |
| [`train.py`](./train.py) | Connect snapshot → rollout → teacher → credit → PPO → checkpoint |
| [`eval.py`](./eval.py) | Evaluate Base/checkpoints without skills and save complete JSONL |
| [`analysis.py`](./analysis.py) | Check evaluation consistency and plot checkpoint comparisons |
| [`skills/alfworld/`](./skills/alfworld) | Pinned general and task-specific skill assets |

Start with the main loop in `train.py`, then follow `rollout.py → teacher.py → advantages.py → loss.py`. This is how a batch of student trajectories becomes a parameter update.
