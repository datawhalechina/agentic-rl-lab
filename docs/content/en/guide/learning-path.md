---
title: An Agentic RL Learning Path from GRPO to Tools and Multimodal Agents
description: Learn policy losses and GRPO, then explore distillation, search, code execution, ALFWorld, multimodal inputs and long-horizon credit assignment.
---

# An Agentic RL learning path

Follow the sequence below or enter through a question you already have. Running one small experiment and changing one variable is often a better way to build intuition than reading every paper at once. If you are setting up for the first time, start with [Environment setup and core concepts](/en/experiments/getting-started/).

## 01 · Understand a policy update

**[Loss Functions](/en/experiments/loss-functions/) → [GRPO](/en/experiments/grpo/) → [DAPO](/en/experiments/dapo/) → [GSPO](/en/experiments/gspo/)**

Begin with log probabilities, importance ratios and advantages. The [GRPO, DAPO and GSPO comparison](./grpo-dapo-gspo) explains what each chapter covers. After completing a GRPO training loop, examine how dynamic sampling, clipping and a sequence-level objective change it.

::: tip Your first change
In the GRPO experiment, keep other parameters fixed and switch among `importance_sampling`, `ppo` and `cispo`. Compare the training behavior. See the [GRPO quick start](/en/experiments/grpo/run) for commands.
:::

## 02 · Turn teacher feedback into a learning signal

**[General OPD](/en/experiments/general-opd/) → [Medical OPD](/en/experiments/medical-opd/) → [OPSD](/en/experiments/opsd/)**

Start with student sampling and teacher scores to understand token-level feedback. Then examine how specialist and general capabilities fit into one training process, before exploring self-distillation.

## 03 · Let a model interact with the world

**[Search-R1](/en/experiments/search-r1/) → [ReTool](/en/experiments/retool/) → [ALFWorld](/en/experiments/alfworld/)**

Introduce search, a Python sandbox and a TextWorld environment in turn. Watch how multi-turn trajectories are stored, how environment observations enter the context, and how training distinguishes model actions from observations.

Then explore [Tournament-GRPO](/en/experiments/tournament-grpo/): when a model's research report has no single correct answer, how can judge comparisons and tournament wins provide rewards?

## 04 · Explore more complex tasks

After GRPO and tool interaction, continue with one of these directions:

| Direction | Reading order | Key question |
| --- | --- | --- |
| Multimodal tool use | [Vision GRPO](/en/experiments/vision-grpo/) → [Spec-o3](/en/experiments/spec-o3/) | How do images and tool outputs enter a training trajectory together? |
| Long-horizon learning and credit assignment | [TEMPO](/en/experiments/tempo/) → [AgentOPSD](/en/experiments/agentopsd/) | Which segment or turn of a long trajectory should receive a learning signal? |
| Fixed-option classification and decisions | [PyTrio-Jev](/en/experiments/pytrio-jev/) | How can one forward pass produce candidate probabilities, and how do we check and calibrate confidence? |

The TEMPO chapter is an algorithm-level reproduction in ALFWorld based on a public technical blog. Each method’s environment, budget and implementation boundaries are documented in its note; read those alongside the results.

## Check your understanding with an experiment

After each chapter, answer four questions:

1. Where does the reward come from, and can it be verified?
2. Which tokens or interaction turns participate in optimization?
3. What variable did this experiment actually change?
4. Which conclusions still need stronger controls?

Continue to [environment setup](./quickstart) or return to [all experiments](/en/experiments/).
