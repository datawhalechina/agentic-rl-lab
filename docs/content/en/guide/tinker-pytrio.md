---
title: Tinker and PyTRIO Compared for LLM Reinforcement Learning
description: Compare Tinker and PyTRIO training workflows, from sampling and rewards to policy updates, tool environments and checkpoints, with links to GRPO and agent RL code.
---

# Tinker and PyTRIO: comparing LLM reinforcement learning workflows

If Tinker introduced you to LLM post-training, you can bring that understanding of the training loop to this lab. The experiments here use PyTRIO. This guide connects the concepts to concrete GRPO, tool-use and multimodal training code.

Tinker here means the training product from Thinking Machines Lab. Its side of this guide is based on official documentation; the PyTRIO side also uses this repository's implementation. **The repository does not yet provide a validated Tinker port or performance benchmarks comparing the two services.** Documentation checked on September 26, 2026.

## Start with the questions shared by a training loop

The [Tinker quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/) covers client creation, sampling, training updates and checkpoints. The [PyTRIO overview](https://docs.pytrio.com/en/docs) also exposes sampling, gradient computation and optimizer updates to your script.

Trace the same questions through either workflow: Which weights generated a response? How was its reward computed? Which tokens contribute to the update? Does the next rollout use the updated weights?

## Connect Tinker concepts to this repository

This table maps reading routes. Check the actual arguments against each SDK.

| Training stage | Tinker documentation entry point | PyTRIO implementation in this lab |
| --- | --- | --- |
| Create a training session | Quickstart: `ServiceClient` and a LoRA training client | Client initialization in the [synchronous GRPO script](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/grpo/01-demo-sync.py) |
| Generate responses with the current policy | Quickstart: sampling and sampler checkpoints | The current-weight sampler, grouped completions and old-policy logprobs in [GRPO](/en/experiments/grpo/) |
| Convert feedback into learning signals | [RL Cookbook](https://tinker-docs.thinkingmachines.ai/cookbook/rl/): rollouts, rewards and advantages | Training data construction in [Loss Functions](/en/experiments/loss-functions/) and [GRPO](/en/experiments/grpo/) |
| Compute gradients and update the policy | `forward_backward` and `optim_step` | GRPO's built-in loss, plus custom objectives in [DAPO](/en/experiments/dapo/) and [GSPO](/en/experiments/gspo/) |
| Organize multi-turn interactions | RL Cookbook: `Env` and `EnvGroupBuilder` | Environment loops in [Search-R1](/en/experiments/search-r1/), [ReTool](/en/experiments/retool/) and [ALFWorld](/en/experiments/alfworld/) |
| Save outputs for later use | Quickstart: training state and sampler weights | Each chapter's checkpoint outputs and quick-start instructions |

Tinker Cookbook organizes environments and trajectory groups into reusable abstractions. This lab presents implementations by task: trace search observations in Search-R1, execution results in ReTool and environment state in ALFWorld. Compare the responsibilities of those objects within the loop first.

## What to check before moving an experiment between SDKs

Start with one prompt and one group of samples before scaling up. This is a suggested validation order; the lab does not currently include a verified cross-SDK conversion script.

1. **Fix the input.** Record the base model, tokenizer, chat template, stop conditions and sampling settings. Different formatting can turn the same text into different training inputs.
2. **Fix the trajectory.** Save generated tokens, rollout logprobs, rewards and advantages, then check their alignment.
3. **Inspect the loss.** Align autoregressive shifting, prompt/tool observation masks, normalization and clipping. Matching method or loss names does not establish identical computation.
4. **Inspect completion boundaries.** Check each SDK's futures and async return values. Confirm that the parameter update has finished before using the corresponding weights.
5. **Check saving and evaluation.** Distinguish inference weights from resumable training state. Compare results on the same test set and budget; handle checkpoint paths, authentication and model access within each service.

The [GRPO quick start](/en/experiments/grpo/run.html) uses a synchronous entry point with an easier-to-trace call order. The [GRPO, DAPO and GSPO guide](./grpo-dapo-gspo.html) explains which parts of the policy update those algorithms change.

## Continue from your goal

- **Run a PyTRIO experiment:** start with the [PyTRIO tutorials](./pytrio.html) and [setup guide](./quickstart.html).
- **Run training on Tinker:** follow the [official Tinker quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/) and [RL Cookbook](https://tinker-docs.thinkingmachines.ai/cookbook/rl/), using this lab's notes as algorithm references.
- **Study agent RL:** read [Search-R1](/en/experiments/search-r1/), [ReTool](/en/experiments/retool/) and [ALFWorld](/en/experiments/alfworld/) to understand how environment feedback enters trajectories.

Model availability, pricing and account permissions change; consult each platform's current official information. This guide helps with workflow reading and migration checks. It does not establish which service is faster or cheaper.
