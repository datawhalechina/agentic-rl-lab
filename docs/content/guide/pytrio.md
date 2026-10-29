---
title: PyTRIO 教程：从 GRPO 到 Agent 强化学习实战
description: 用 PyTRIO 学习 LLM 后训练：从安装登录和 GSM8K GRPO 出发，阅读 DAPO、GSPO、工具调用、多模态与蒸馏的代码和实验记录。
---

# PyTRIO 教程：从 GRPO 到 Agent 强化学习实战

我用 PyTRIO 把论文里的训练方法拆成可以运行、修改和检查的实验。这份教程入口把本仓库的代码与笔记串起来：先完成一次 GRPO 训练，再逐步加入自定义 loss、工具环境和图片输入。

PyTRIO 将模型训练与采样放到远程服务，本地脚本组织数据、奖励和实验循环。服务能力与接口说明见 [PyTRIO 官方文档](https://docs.pytrio.com/docs)。这里记录的是我在具体任务中的实现、观察和边界。

## 第一次使用 PyTRIO，从哪里开始？

1. 按[安装与首次训练指南](./quickstart.html)获取仓库、安装锁定依赖，并登录 PyTRIO 与 SwanLab。
2. 阅读 [GRPO 原理与代码](/experiments/grpo/)，先认清 prompt、group、reward 和 advantage 在代码里的位置。
3. 按 [GRPO 快速启动](/experiments/grpo/run.html)运行 GSM8K 小规模训练。当前指南使用 `experiments/grpo/01-demo-sync.py` 同步入口。
4. 检查训练日志里的 reward、退化组比例、生成长度与训练 token 数，并保存程序输出的 sampler weights 路径。

仓库当前锁定 `pytrio==0.2.9`，要求 Python 3.13 或以上。复现实验时优先使用仓库的锁定环境；想尝试其他 SDK 版本时，单独检查接口和实验结果。

## 按任务选择教程

| 你想解决的问题 | 阅读与运行入口 | 阅读时重点检查 |
| --- | --- | --- |
| 完成一次数学推理 RL 训练 | [GRPO](/experiments/grpo/) · [快速启动](/experiments/grpo/run.html) | 同题组采样、奖励、旧策略 logprob 与 advantage |
| 改变策略更新方式 | [Loss Functions](/experiments/loss-functions/) · [DAPO](/experiments/dapo/) · [GSPO](/experiments/gspo/) | 裁剪、动态采样和序列级目标各自改了哪一步 |
| 使用教师反馈训练学生 | [General OPD](/experiments/general-opd/) · [Medical OPD](/experiments/medical-opd/) · [OPSD](/experiments/opsd/) | 学生轨迹、教师打分与 token 级学习信号 |
| 训练会调用工具的 Agent | [Search-R1](/experiments/search-r1/) · [ReTool](/experiments/retool/) | 搜索或代码执行结果怎样进入上下文，哪些 token 参与 loss |
| 研究长程环境交互 | [ALFWorld](/experiments/alfworld/) · [TEMPO](/experiments/tempo/) · [AgentOPSD](/experiments/agentopsd/) | 环境状态、轨迹预算、终局奖励与信用分配 |
| 加入图片和视觉工具 | [Vision GRPO](/experiments/vision-grpo/) · [Spec-o3](/experiments/spec-o3/) | 图文输入、视觉 token 对齐和固定评测条件 |
| 训练固定选项的决策模型 | [PyTrio-Jev](/experiments/pytrio-jev/) · [快速启动](/experiments/pytrio-jev/run.html) | marker 位置监督、候选概率与温度校准 |

GSPO 笔记会区分核心 loss 实现与完整训练复现；其他章节也保留各自的环境、预算和实现边界。阅读结果时，请一起看这些条件。

## 读 PyTRIO 训练代码时，先找这五个位置

以仓库的 [GRPO 同步脚本](https://github.com/datawhalechina/agentic-rl-lab/blob/main/experiments/grpo/01-demo-sync.py)为例：

1. `ServiceClient` 与 `create_lora_training_client`：选择基模并创建训练会话。
2. `save_weights_and_get_sampling_client`：从当前训练权重取得 sampler，再做组采样。
3. reward 与 advantage 的计算：追踪它们如何对应到每条 completion。
4. `forward_backward` 或 `forward_backward_custom`，随后 `optim_step`：定位 loss、梯度与参数更新。
5. `save_weights_for_sampler`：找到训练结束后的推理权重输出。

先沿着这些位置读通一次数据流，再修改一个变量。例如只切换 loss，保持模型、数据、采样参数和训练预算一致，观察日志发生了什么变化。

## 常见问题

### 本地没有 GPU，也能运行吗？

本仓库把模型采样与训练交给 PyTRIO 服务，本地仍需运行数据处理、实验循环以及所选任务的工具或环境。需要可用的服务账户和训练资源；“本地不跑模型训练”不代表实验没有资源成本。

### 训练 reward 上升，就能说明模型变好了吗？

训练日志只能说明当前训练样本上的行为。比较模型能力时，还要固定测试集、评测协议与采样预算。GRPO 章节当前没有独立 `eval.py`，请保留这个边界；其他任务按各自文档运行评测。

### 我熟悉 Tinker，这些笔记能怎么用？

从 [Tinker 与 PyTRIO 的训练工作流对照](./tinker-pytrio.html)开始，把采样、奖励、策略更新与 checkpoint 对应到具体实验。当前仓库运行入口使用 PyTRIO，迁移到其他 SDK 需要单独适配和验证。

下一步：[安装 PyTRIO 并跑通第一个实验](./quickstart.html)，或查看[完整学习路线](./learning-path.html)。
