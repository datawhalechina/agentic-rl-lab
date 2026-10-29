---
title: PyTRIO Tutorials for GRPO and Agent Reinforcement Learning
description: Learn LLM post-training with PyTRIO, from setup and GRPO on GSM8K to DAPO, GSPO, tool use, multimodal training and distillation, with code and experiment notes.
---

# PyTRIO tutorials for GRPO and agent reinforcement learning

I use PyTRIO to turn methods from papers into experiments that I can run, modify and inspect. This guide connects the repository's code and notes: complete a GRPO run first, then explore custom losses, tool environments and image inputs.

PyTRIO runs model training and sampling remotely while local scripts organize data, rewards and the experiment loop. See the [official PyTRIO documentation](https://docs.pytrio.com/en/docs) for service capabilities and API details. This lab documents implementations, observations and limitations for specific tasks.

## Where should a new PyTRIO user start?

1. Follow the [setup and first-run guide](./quickstart.html) to get the repository, install its locked dependencies, and log in to PyTRIO and SwanLab.
2. Read the [GRPO walkthrough](/en/experiments/grpo/) and locate prompts, groups, rewards and advantages in the code.
3. Use the [GRPO quick start](/en/experiments/grpo/run.html) for a small GSM8K run. The current guide uses the synchronous `experiments/grpo/01-demo-sync.py` entry point.
4. Inspect reward, the fraction of degenerate groups, completion length and training token counts. Keep the sampler weights path printed by the script.

The repository currently pins `pytrio==0.2.9` and requires Python 3.13 or later. Use the locked environment for reproduction; check APIs and experimental results separately when trying another SDK version.

## Choose a tutorial by task

| Your question | Reading and running entry points | What to inspect |
| --- | --- | --- |
| How do I run mathematical reasoning RL? | [GRPO](/en/experiments/grpo/) · [Quick start](/en/experiments/grpo/run.html) | Groups from the same prompt, rewards, old-policy logprobs and advantages |
| How do policy update methods differ? | [Loss Functions](/en/experiments/loss-functions/) · [DAPO](/en/experiments/dapo/) · [GSPO](/en/experiments/gspo/) | Where clipping, dynamic sampling and sequence-level objectives change the loop |
| How can teacher feedback train a student? | [General OPD](/en/experiments/general-opd/) · [Medical OPD](/en/experiments/medical-opd/) · [OPSD](/en/experiments/opsd/) | Student trajectories, teacher scores and token-level learning signals |
| How do I train an agent to use tools? | [Search-R1](/en/experiments/search-r1/) · [ReTool](/en/experiments/retool/) | How search or code results enter context, and which tokens contribute to loss |
| How do agents learn across long interactions? | [ALFWorld](/en/experiments/alfworld/) · [TEMPO](/en/experiments/tempo/) · [AgentOPSD](/en/experiments/agentopsd/) | Environment state, trajectory budgets, terminal rewards and credit assignment |
| How do images and visual tools enter training? | [Vision GRPO](/en/experiments/vision-grpo/) · [Spec-o3](/en/experiments/spec-o3/) | Multimodal inputs, visual token alignment and fixed evaluation conditions |
| How do I train a model for fixed-option decisions? | [PyTrio-Jev](/en/experiments/pytrio-jev/) · [Quick start](/en/experiments/pytrio-jev/run.html) | Marker-position supervision, candidate probabilities and temperature calibration |

The GSPO note distinguishes its core-loss implementation from a complete training reproduction. Every chapter has its own environment, budget and implementation limits; read them alongside the results.

## Five places to find in PyTRIO training code

In this repository's [synchronous GRPO script](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/grpo/01-demo-sync.py), trace:

1. `ServiceClient` and `create_lora_training_client`: base-model selection and the training session.
2. `save_weights_and_get_sampling_client`: a sampler from the current training weights, followed by group sampling.
3. Reward and advantage computation: how each value maps to a completion.
4. `forward_backward` or `forward_backward_custom`, followed by `optim_step`: the loss, gradients and parameter update.
5. `save_weights_for_sampler`: the inference weights saved at the end.

Follow one batch through these stages before changing one variable. For example, change only the loss while keeping the model, data, sampling settings and training budget fixed, then inspect the logs.

## Common questions

### Can I run this without a local GPU?

Model sampling and training run on the PyTRIO service. Your local machine still runs data processing, the experiment loop and any tools or environments the task needs. You need service access and training resources; remote model execution does not make an experiment free of resource costs.

### Does a higher training reward prove the model improved?

Training logs describe behavior on training samples. Capability comparisons also need a fixed test set, evaluation protocol and sampling budget. The GRPO chapter currently has no separate `eval.py`; preserve that limitation. Follow each other task's own evaluation instructions.

### I know Tinker. How can I use these notes?

Start with the [Tinker and PyTRIO workflow comparison](./tinker-pytrio.html) to connect sampling, rewards, policy updates and checkpoints to concrete experiments. This repository uses PyTRIO; another SDK needs its own adaptation and validation.

Next: [set up PyTRIO and run your first experiment](./quickstart.html), or follow the [complete learning path](./learning-path.html).
