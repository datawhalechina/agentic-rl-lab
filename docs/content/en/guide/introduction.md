---
title: About Agentic RL Lab
description: An open research notebook connecting LLM and agent reinforcement learning papers with code, training records, evaluations and honest implementation limits.
---

# About Agentic RL Lab

**Agentic RL Lab** is an open repository of experiment notes and tutorials.

I use PyTRIO to reproduce and study LLM and agent reinforcement learning methods, bringing together the algorithms, runnable code, training process and evaluation. Each note should give you something concrete to change, run and check for yourself.

## Author and sources

This lab is maintained by [KMnO4-zx](https://github.com/KMnO4-zx). Notes link to their papers, implementation and public experiment records so that readers can trace the evidence behind a conclusion. The Chinese and English editions share the same experiment code and original figures. Original code comments, logs and text inside figures may remain in Chinese. Report translation or reproduction issues through GitHub Issues or the edit link on each page.

## What an experiment note contains

Between a paper and a working experiment lie many practical details: how sampling is organized, how rewards are computed, which tokens receive updates, and how evaluation differs from training.

The notes follow these questions through three stages:

1. **Paper and intuition:** the problem, the method and its core variables.
2. **Code and training:** how data, rewards, losses and the training loop fit together.
3. **Results and limitations:** what was observed, and where the implementation differs from the original method.

Each chapter records its own models, data and configuration. Scores from different chapters are not directly comparable. Algorithm-level reproductions also make different simplifications; read each note’s limitations alongside its results.

## Why PyTRIO?

I wanted to study Agentic RL in depth, but limited GPU access and the complexity of training and inference infrastructure kept getting in the way. PyTRIO lets a remote service handle model training and sampling while I keep the experiment loop local and easier to inspect and change.

Experiments use [SwanLab](https://swanlab.cn/) for tracking. Relevant public records and paper references appear in the individual notes so readers can compare the implementation with the observations.

## Where to start

- New to the topic? Read the [learning path](./learning-path), then complete [environment setup](./quickstart).
- Already familiar with training? Choose a research direction in the [experiment handbook](/en/experiments/).
- Ready to run code? Open a chapter’s quick start and follow its dependencies and commands.

## Improve the experiments together

Use [GitHub Issues](https://github.com/datawhalechina/agentic-rl-lab/issues) to share reproduction notes, report problems or propose experiments. Include the model name, command, dependency versions and relevant logs to make a report easier to investigate.

See the [GitHub repository](https://github.com/datawhalechina/agentic-rl-lab) for source code and license information. The WeChat community entry is in [this issue](https://github.com/datawhalechina/agentic-rl-lab/issues/13).
