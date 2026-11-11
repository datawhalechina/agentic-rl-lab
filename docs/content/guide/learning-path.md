---
title: Agentic RL 学习路线：从 GRPO 到工具调用与多模态
description: 从损失函数和 GRPO 开始，逐步探索蒸馏、工具交互和多模态强化学习。
---

# Agentic RL 学习路线

可以按下面的顺序走，也可以带着自己的问题进入某个章节。先跑通一个小实验，再改一个变量，通常比一口气读完所有论文更容易建立直觉。第一次配置环境的话，从[环境配置与基础概念](/experiments/getting-started/)开始。

## 01 · 理解一次策略更新

**[Loss Functions](/experiments/loss-functions/) → [GRPO](/experiments/grpo/) → [DAPO](/experiments/dapo/) → [GSPO](/experiments/gspo/)**

先理解 logprob、重要性比率与 advantage。可以先用[GRPO、DAPO 与 GSPO 对照](./grpo-dapo-gspo)了解这三个章节各自覆盖的内容。用 GRPO 走通一次训练循环后，再观察动态采样、裁剪与序列级目标分别改变了什么。

::: tip 第一个可以动手的改动
在 GRPO 中保持其他参数一致，切换 `importance_sampling`、`ppo` 与 `cispo`，对照训练行为。具体运行入口见 [GRPO 快速启动](/experiments/grpo/run)。
:::

## 02 · 把教师反馈变成学习信号

**[General OPD](/experiments/general-opd/) → [Medical OPD](/experiments/medical-opd/) → [OPSD](/experiments/opsd/)**

从 Student 采样与 Teacher 打分开始，理解 token 级反馈。接着看专业能力与通用能力怎样组织在同一次训练里，再探索 self-distillation 的实现。

## 03 · 让模型与外部世界交互

**[Search-R1](/experiments/search-r1/) → [ReTool](/experiments/retool/) → [ALFWorld](/experiments/alfworld/)**

依次接入搜索、Python 沙箱与 TextWorld 环境。重点观察多轮轨迹如何保存、环境返回如何进入上下文，以及训练时怎样区分模型动作与环境观察。

接着可以看 [Tournament-GRPO](/experiments/tournament-grpo/)：当模型搜索后写出的研究报告没有标准答案时，怎样用裁判比较和淘汰赛胜场构造奖励？

## 04 · 探索更复杂的任务

已经理解 GRPO 与工具交互后，可以从下面几个方向继续：

| 研究方向 | 阅读顺序 | 重点问题 |
| --- | --- | --- |
| 多模态工具使用 | [Vision GRPO](/experiments/vision-grpo/) → [Spec-o3](/experiments/spec-o3/) | 图片与工具结果怎样共同进入训练轨迹？ |
| 长程学习与信用分配 | [TEMPO](/experiments/tempo/) → [AgentOPSD](/experiments/agentopsd/) | 长轨迹中的哪一段、哪一轮应该得到学习信号？ |
| 固定选项的分类决策 | [PyTrio-Jev](/experiments/pytrio-jev/) | 怎样用一次前向得到候选概率，再检查和校准置信度？ |

TEMPO 章节是基于公开技术博客、在 ALFWorld 中进行的算法级复现。每个方法的环境、预算与实现边界都在正文中说明，阅读结果时也请一起查看。

## 用实验检验自己的理解

每完成一篇，试着回答四个问题：

1. 奖励从哪里来，能否验证？
2. 哪些 token 或交互轮次参与了优化？
3. 这次实验实际改变了什么变量？
4. 哪些结论还需要更充分的对照？

准备好后，进入[环境准备](./quickstart)，或返回[全部实验](/experiments/)。
