---
title: 关于这个 Lab
description: 一个从论文、代码和真实实验出发的 Agent 强化学习开源笔记。
---

# 关于这个 Lab

这里是 **Agentic RL Lab**，一个偏实验记录和教程的开源仓库。

我会用 PyTRIO 复现和拆解 LLM / Agent 强化学习方法，把算法、可运行代码、训练过程和评测放在一起。希望每读完一篇，你都能找到一个可以亲手改动、运行和验证的入口。

## 什么是 Agentic RL

一般来说，**Agentic RL 是指把 LLM 当作策略，放进一个有状态、有时间跨度的环境里，用强化学习优化它在多轮执行中的「观测—动作—反馈」行为。** 其中「动作」是文本或工具调用，「观测」来自环境（搜索结果、代码执行输出、游戏状态等）；只有模型自己生成的 token 参与 loss，而奖励通常来自任务的成败，而不只是答案对不对。

它和单轮 RLVR（例如在 GSM8K 上跑 GRPO）共用同一套 loss 与 advantage，区别在轨迹：

| | 单轮 RLVR | Agentic RL |
| --- | --- | --- |
| 状态 | prompt 加上已生成的 token | 再加一层模型看不见、只能通过工具感知的环境状态 |
| 一次 rollout | 生成一次 | 生成多次，中间穿插工具调用 |
| 参与 loss 的 token | 全部 | 只有模型的动作，环境返回的 observation 要 mask 掉 |
| 奖励 | 答案对错 | 任务成败，通常更稀疏、更延迟 |

所以这个 Lab 的笔记中真正多轮的是 [Search-R1](/experiments/search-r1/)、[ReTool](/experiments/retool/)、[ALFWorld](/experiments/alfworld/)、[TEMPO](/experiments/tempo/)、[AgentOPSD](/experiments/agentopsd/) 与 [Spec-o3](/experiments/spec-o3/)；而[GRPO](/experiments/grpo/)、[DAPO](/experiments/dapo/)、[GSPO](/experiments/gspo/)、[OPD](/experiments/general-opd/)、[OPSD](/experiments/opsd/) 是它们的前置技术。学习路线从单轮讲起，是因为它是多轮训练的最小底座：先学习同一套 loss、同一套采样，再把 rollout 拉长、逐步把环境接进来。学术界的定义与分类可以参考综述 [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](https://arxiv.org/abs/2509.02547)。

## 作者与内容来源

这个 Lab 由 [KMnO4-zx](https://github.com/KMnO4-zx) 维护。每篇笔记尽量保留论文出处、代码入口、训练记录和评测条件，方便读者追查结论。中英文版本共用实验代码与原始结果图；代码中的原始注释、日志和图内文字可能保留中文。发现翻译或复现问题时，可以通过页面底部的编辑链接或 GitHub Issues 反馈。

### 内容贡献者

- [MRLE](https://github.com/mrle0429)（北京交通大学）：[Tournament-GRPO](/experiments/tournament-grpo/)

## 每篇实验里有什么

一篇论文从提出想法到真正跑起来，中间往往藏着不少细节：采样怎样组织，奖励怎样计算，哪些 token 参与更新，评测又和训练有什么区别。

这里的笔记沿着这些问题展开：

1. **论文与直觉**：方法在解决什么问题，核心变量是什么。
2. **代码与训练**：数据、reward、loss、训练循环怎样连接起来。
3. **结果与边界**：记录实际观察，以及当前实现和原方法之间的差异。

正文中的训练记录保留各自的模型、数据和配置。不同章节的分数不能直接放在同一张榜单里比较；算法级实现也有各自的简化，请结合每篇文末的限制阅读。

## 为什么使用 PyTRIO

我曾经很想深入研究 Agentic RL，但没卡、训推工程复杂等问题让我迟迟没有动手。通过 PyTRIO，把模型训练和采样交给远程服务，本地保留更容易理解和改动的实验循环，我才开始把这些想法一个个跑起来。

实验过程使用 [SwanLab](https://swanlab.cn/) 记录。相关公开记录与论文引用放在各篇文章中，方便对照实现与观察。

## 从哪里开始

- 第一次接触：先看[学习路线](./learning-path)，再完成[环境准备](./quickstart)。
- 已经有训练经验：在[实验手册](/experiments/)中按研究方向选一篇。
- 想直接运行：进入章节的「快速启动」，按该章节的依赖和命令操作。

## 一起把实验做好

欢迎在 [GitHub Issues](https://github.com/datawhalechina/agentic-rl-lab/issues) 分享复现记录、报告问题或提出新的实验想法。提交问题时，附上模型名称、运行命令、依赖版本与必要日志，会更容易定位。

完整源代码与许可说明见 [GitHub 仓库](https://github.com/datawhalechina/agentic-rl-lab)。微信交流群入口在[这个 Issue](https://github.com/datawhalechina/agentic-rl-lab/issues/13)。
