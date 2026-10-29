---
title: PyTRIO 安装与入门：跑通第一个 GRPO 实验
description: 安装 PyTRIO 实验依赖、登录训练服务与 SwanLab，并在 GSM8K 上启动首个 GRPO 训练，了解本地环境和远程资源要求。
---

# PyTRIO 安装与入门：跑通第一个 GRPO 实验

本地负责数据处理、工具执行和训练循环，模型采样与训练通过 PyTRIO 远程执行。开始前，需要 Python **3.13 或以上**、`uv`，以及可用的 PyTRIO 账户和训练资源。

想先了解各个实验的用途，可以阅读 [PyTRIO 教程与实战路线](./pytrio.html)；熟悉 Tinker 的读者可以从[训练工作流对照](./tinker-pytrio.html)进入。

## 获取代码与安装依赖

```bash
git clone https://github.com/KMnO4-zx/agentic-rl-lab.git
cd agentic-rl-lab
uv sync
```

`pyproject.toml` 和 `uv.lock` 记录实验依赖。不同章节对数据集、外部 API 和环境的要求可能不同，请以该章节的快速启动为准。

## 登录服务

```bash
uv run trio login
uv run swanlab login
```

PyTRIO 负责训练与采样，SwanLab 用来记录训练指标。服务账户与可用模型请以各平台当前配置为准。

## 运行 GRPO

从 [GRPO 快速启动](/experiments/grpo/run) 开始。该页面直接来自仓库的 `experiments/grpo/start.md`，包含当前推荐入口、采样参数和指标说明。

::: tip 先认识整个循环
配合阅读 [GRPO 算法拆解](/experiments/grpo/)，对照「准备 prompt → 组采样 → 奖励与 advantage → 更新策略」四个步骤。训练调用会消耗远程服务资源，先按快速指南的小规模配置运行。
:::

## 进入交互环境

ALFWorld、TEMPO 和 AgentOPSD 还需要环境依赖：

```bash
uv sync --extra alfworld
```

随后按对应章节准备环境与数据：[ALFWorld](/experiments/alfworld/run)、[TEMPO](/experiments/tempo/)、[AgentOPSD](/experiments/agentopsd/run)。Search-R1 的检索后端、ReTool 的本地沙箱和 Spec-o3 的多模态数据也各有独立配置。

## 记录一次可复查的实验

保留模型名称、checkpoint、数据划分、随机种子、采样预算和评测命令。对比实验时，先对齐这些条件，再讨论方法的效果。

可以从[实验手册](/experiments/)选择下一个方向。
