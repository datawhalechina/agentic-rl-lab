---
title: PyTRIO Setup and Your First GRPO Training Run
description: Install the PyTRIO experiment dependencies, log in to the training service and SwanLab, and launch GRPO on GSM8K with the required local and remote resources.
---

# PyTRIO setup and your first GRPO training run

Your local machine handles data, tools and the experiment loop. PyTRIO executes model sampling and training remotely. You need **Python 3.13 or later**, `uv`, and a PyTRIO account with access to training resources.

For a map of the experiments, read the [PyTRIO tutorials](./pytrio.html). Readers familiar with Tinker can start with the [training workflow comparison](./tinker-pytrio.html).

## Get the code and install dependencies

```bash
git clone https://github.com/datawhalechina/agentic-rl-lab.git
cd agentic-rl-lab
uv sync
```

`pyproject.toml` and `uv.lock` record the experiment dependencies. Dataset, external API and environment requirements vary by chapter; follow the relevant quick start.

## Log in to the services

```bash
uv run trio login
uv run swanlab login
```

PyTRIO handles training and sampling; SwanLab tracks training metrics. Available models and account access depend on the platforms’ current configuration.

## Run GRPO

Start with the [GRPO quick start](/en/experiments/grpo/run). It translates the repository’s `experiments/grpo/start.md`, including the current entry point, sampling settings and metric descriptions.

::: tip Understand the loop
Read the [GRPO walkthrough](/en/experiments/grpo/) alongside the code: prepare prompts → sample a group → compute rewards and advantages → update the policy. Training consumes remote service resources; begin with the small configuration in the quick start.
:::

## Add interactive environments

ALFWorld, TEMPO and AgentOPSD also need the environment dependencies:

```bash
uv sync --extra alfworld
```

Then prepare data and environments for [ALFWorld](/en/experiments/alfworld/run), [TEMPO](/en/experiments/tempo/) or [AgentOPSD](/en/experiments/agentopsd/run). Search-R1 retrieval backends, the ReTool local sandbox and Spec-o3 multimodal data have their own configuration requirements.

## Keep an experiment reproducible

Record the model name, checkpoint, data split, random seed, sampling budget and evaluation commands. Align these conditions before comparing methods.

Choose your next direction in the [experiment handbook](/en/experiments/).
