---
title: 关于这个 Lab
description: 一个从论文、代码和真实实验出发的 Agent 强化学习开源笔记。
---

# 关于这个 Lab

这里是 **Agentic RL Lab**，一个偏实验记录和教程的开源仓库。

我会用 PyTRIO 复现和拆解 LLM / Agent 强化学习方法，把算法、可运行代码、训练过程和评测放在一起。希望每读完一篇，你都能找到一个可以亲手改动、运行和验证的入口。

## 什么是 Agentic RL

Agentic RL 训练的是模型完成一次任务的过程，而不只是一个回答。LLM 是策略，环境是有状态的外部系统：搜索引擎、Python 解释器或文本游戏。每一轮，模型根据当前上下文生成一段文本或一个工具调用作为动作；环境执行它、更新自己的状态，再把结果作为 observation 返回。这个过程一直重复到任务结束。

一条轨迹可以写成：

```text
prompt → action_1 → observation_1 → action_2 → ... → action_T → reward
```

这条轨迹就是一次 rollout。模型能看到的是 prompt、自己此前的动作，以及环境返回的 observation；环境内部的状态它看不到，只能通过动作去探查。下一步能观察到什么，取决于这一步做了什么——轨迹由策略和环境共同决定，用 RL 的话说，这是一个部分可观测的决策过程（POMDP）。

训练时，整条轨迹被拼成一个 token 序列送进模型。prompt 和单轮时一样不参与 loss；多轮轨迹里环境返回的 observation 也是给定输入，同样要 mask 掉。真正参与 loss 的只有模型自己生成的动作 token（推理、工具调用、最终回答）。奖励通常在轨迹结束时才出现，再由 GRPO、PPO 这类算法转成 advantage 回传。

Agentic RL 和单轮 RLVR（例如在 GSM8K 上跑 GRPO）共用同一套 loss 与 advantage，区别集中在轨迹：

| | 单轮 RLVR | Agentic RL |
| --- | --- | --- |
| 状态 | prompt 加上已生成的 token | 再加一层模型看不见、只能通过工具感知的环境状态 |
| 一次 rollout | 生成一次 | 生成多次，中间穿插工具调用 |
| 参与 loss 的 token | 模型生成的全部 token（prompt mask） | 只有模型的动作 token（prompt 和 observation 都 mask） |
| 奖励 | 答案对错 | 任务成败，通常更稀疏、更延迟 |

轨迹变长之后，训练要额外处理三件事：

- **信用分配**：奖励在终点才出现，组内相对 advantage 无法指出中间哪一步真正起了作用。[TEMPO](/experiments/tempo/) 把长轨迹切成小段，训练生成式 critic 在段末估值；[AgentOPSD](/experiments/agentopsd/) 让 teacher 重看整条轨迹，为每个 turn 补监督。
- **上下文就是状态**：环境状态不会自己进入模型，只能被 observation 带进上下文。保留多少历史、什么时候截断，会直接影响模型能学到什么。
- **采样成本**：每一步动作都可能调用外部服务，rollout 比单轮慢，也更容易失败。

Agentic RL Lab 的多轮章节从不同环境切入：[Search-R1](/experiments/search-r1/) 接搜索，[ReTool](/experiments/retool/) 接代码执行，[ALFWorld](/experiments/alfworld/) 接文本游戏，[Spec-o3](/experiments/spec-o3/) 把图片和工具返回一起放进轨迹。单轮的 [GRPO](/experiments/grpo/)、[DAPO](/experiments/dapo/)、[GSPO](/experiments/gspo/) 把采样、reward、advantage 与策略更新连成完整的训练循环，[OPD](/experiments/general-opd/)、[OPSD](/experiments/opsd/) 则把 teacher 的反馈变成学习信号。学术界的定义与分类可以参考综述 [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](https://arxiv.org/abs/2509.02547)。

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
