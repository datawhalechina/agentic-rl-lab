---
title: "Contribute: Help Lower the Barrier to Agentic RL"
description: Contributing guide for Agentic RL Lab — how to propose a new algorithm reproduction chapter, chapter structure and quality requirements, engineering conventions, and the full flow from Issue proposal to merged PR.
---

# Contribute

This repo has a single goal: **to let more people reproduce Agentic RL algorithms and papers at a much lower cost, and actually get hands-on with Agentic RL research** — instead of being locked out by the lack of an 8-GPU node, by the complexity of unified training-and-inference stacks, or by the tightly coupled engineering of frameworks like Verl.

Thanks to [PyTRIO](https://pytrio.com), every reproduction in this repo achieves this: for the price of a cup of bubble tea, in about twenty training steps, without owning a single consumer GPU, you can see real training progress. We believe the value of algorithm research lies in understanding the algorithm itself, not in spending three months wrestling with infrastructure first. Every chapter you contribute helps more people like us cross that threshold.

This page mirrors the [Contributing Guide](https://github.com/datawhalechina/agentic-rl-lab/blob/main/CONTRIBUTING_EN.md) on GitHub.

## Ways to Contribute

| Type | Process |
| --- | --- |
| New algorithm chapter | Open an Issue proposal first → maintainer approves the topic and directory name → submit a PR → review and merge |
| Bug fixes / additional experiments for existing chapters | Submit a PR directly; describe the problem and how to reproduce it |
| Typos, docs improvements, translations | Submit a PR directly |

## Full Process for a New Algorithm Chapter

1. **Open an Issue proposal**. Use the [Algorithm Proposal template](https://github.com/datawhalechina/agentic-rl-lab/issues/new/choose) and include:
   - Which paper / algorithm you want to reproduce (with a link to the paper)
   - Which dataset and base model you plan to use
   - Your design for the reward / advantage / loss training loop
   - Estimated PyTRIO cost (for reference: a single training run in existing chapters usually costs a few dozen to a few hundred CNY)
2. **Wait for confirmation**. The maintainer will align with you on whether the topic fits and whether it conflicts with ongoing work, and confirm the directory name under `experiments/<algorithm-name>/` (lowercase letters and hyphens, without a chapter number).
3. **Fork and develop**. Suggested branch name: `feat/algorithm-name`.
4. **Submit the PR**. Link the Issue in the PR description, and attach training records and evaluation results (see Quality Requirements below).
5. **Review**. The maintainer reviews from two angles — "is the algorithm explained clearly" and "can the code actually be reproduced" — and may ask for a few rounds of revision. This is a normal part of the process.

## Chapter Structure Conventions

New chapters should strictly follow the structure of existing chapters (use `experiments/search-r1/` as the reference template):

```text
experiments/algorithm-name/
├── readme.md          # Main chapter text (long-form, blog style)
├── start.md           # Quick start: full commands for data prep → training → evaluation
├── data.py / prepare_data.py   # Data download and preprocessing
├── reward.py          # Reward computation
├── rollout.py         # Trajectory sampling (if applicable)
├── train.py           # Training entry point
├── eval.py            # Evaluation entry point
├── analysis.py        # Result analysis and plotting
├── images/            # Images referenced by the readme (including the cover)
└── .env.example       # Config template if API keys are needed
```

- All scripts run from the **repo root**, e.g. `uv run python experiments/algorithm-name/train.py ...`.
- For the style of `readme.md`, follow the existing chapters: first explain where the algorithm comes from, what problem it solves, and what its core variables are; then explain how to reproduce it. Intuition first; everyday analogies (bubble tea, Heytea) are encouraged.
- `start.md` must let a new reader copy the commands in order and get a working run: login (`trio login`, `swanlab login`) → data → training → evaluation.

## Quality Requirements (Merge Bar)

1. **Actually runs**: the code must have completed a full end-to-end training run on PyTRIO — not just import checks or compilation. Please attach a SwanLab training record link in the PR.
2. **Evidence of effect**: on a fixed evaluation set, report Base Model vs. trained checkpoint numbers (e.g. EM / Format Rate), and describe the evaluation setup.
3. **Cost transparency**: state the actual cost (in CNY) and wall-clock time of a single training run, so readers can judge whether they can afford to reproduce it.
4. **Low barrier first**: prefer configurations where a small model and a small number of steps already show metric movement. The spirit of this repo is that a learner should be able to confirm "the code works" within 20 steps.
5. **Honest boundaries**: state clearly which conclusions a single small-scale run can support, and which it cannot.

## Engineering Conventions

- Dependencies are managed with [uv](https://docs.astral.sh/uv/); Python `>=3.13` is required. Before adding a dependency, check that `pyproject.toml` has no equivalent already.
- **Do not commit**: `.env` or any secrets, dataset files, `swanlog/`, evaluation outputs, or checkpoints. Add dataset and evaluation output directories to `.gitignore`; put required keys in an `.env.example` template with empty values.
- Use [SwanLab](https://swanlab.cn/) for experiment tracking.
- Images live in the chapter's `images/` directory and are referenced with relative paths.

## Language

Chapter text is primarily in Chinese (consistent with existing chapters); if an English version is more convenient for you, feel free to provide both. Code comments and variable names should be in English.

## Community

- WeChat group: see [Issue #13](https://github.com/datawhalechina/agentic-rl-lab/issues/13)
- Not sure whether a topic fits? Open an Issue and ask, or chat in the group — you don't need a complete plan before reaching out.

Looking forward to your contribution — let's lower the barrier to Agentic RL together!
