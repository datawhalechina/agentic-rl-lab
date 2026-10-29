---
title: Tinker 与 PyTRIO：LLM 强化学习训练工作流对照
description: 面向 Tinker 和 PyTRIO 读者的强化学习指南：对照采样、奖励、策略更新、工具环境与 checkpoint，并找到 GRPO 和 Agent RL 的代码入口。
---

# Tinker 与 PyTRIO：LLM 强化学习训练工作流对照

如果你从 Tinker 开始了解 LLM 后训练，可以带着熟悉的训练循环阅读这个 Lab。这里的实验使用 PyTRIO；本文把两边的概念对应起来，帮助你找到 GRPO、工具调用和多模态训练的具体代码。

本文中的 Tinker 指 Thinking Machines Lab 的训练产品。Tinker 部分依据其官方文档整理，PyTRIO 部分结合本仓库的实现。**当前仓库尚未提供经过验证的 Tinker 移植版本，也没有两种服务的性能对比实验。** 文档核对日期：2026-09-26。

## 先认识训练循环的共同问题

Tinker 的[官方快速开始](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)展示了客户端创建、采样、训练更新和 checkpoint 操作。PyTRIO 的[官方概览](https://docs.pytrio.com/docs)也将采样、梯度计算与优化器更新暴露为脚本可以调用的操作。

读这类训练代码时，可以沿着同一组问题往下追：当前用哪份权重生成回答？reward 怎样计算？哪些 token 参与更新？下一轮采样是否使用更新后的权重？

## 从 Tinker 的概念找到本仓库的实现

下面是阅读路线的对应，具体调用参数仍应按各自 SDK 检查。

| 训练环节 | Tinker 官方文档中的入口 | 本仓库的 PyTRIO 阅读入口 |
| --- | --- | --- |
| 创建训练会话 | Quickstart 中的 `ServiceClient` 与 LoRA training client | [GRPO 同步脚本](https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/experiments/grpo/01-demo-sync.py)的 client 初始化 |
| 用当前策略生成回答 | Quickstart 的 sampling 与 sampler checkpoint | [GRPO](/experiments/grpo/)中的当前权重 sampler、组采样与旧策略 logprob |
| 把反馈变成更新信号 | [RL Cookbook](https://tinker-docs.thinkingmachines.ai/cookbook/rl/)的 rollout、reward 和 advantage | [Loss Functions](/experiments/loss-functions/)与 [GRPO](/experiments/grpo/)的训练数据构造 |
| 计算梯度并更新策略 | `forward_backward` 与 `optim_step` | GRPO 的内置 loss，以及 [DAPO](/experiments/dapo/)和 [GSPO](/experiments/gspo/)的自定义目标 |
| 组织多轮环境交互 | RL Cookbook 中的 `Env` 与 `EnvGroupBuilder` | [Search-R1](/experiments/search-r1/)、[ReTool](/experiments/retool/)和 [ALFWorld](/experiments/alfworld/)的环境循环 |
| 保存结果供后续使用 | Quickstart 的训练状态与采样权重保存 | 各章节的 checkpoint 输出与快速启动说明 |

Tinker Cookbook 将环境与同组轨迹组织成可复用的抽象。本仓库按任务展示实现：读 Search-R1 时追踪搜索 observation，读 ReTool 时追踪代码执行结果，读 ALFWorld 时追踪环境状态。比较时先对齐这些对象在训练循环里的职责。

## 如果想把实验从一种 SDK 迁移到另一种

建议先拿一个 prompt 和一个采样组做对应检查，再扩大训练规模。这是迁移时的检查顺序，当前没有附带已验证的跨 SDK 转换脚本。

1. **固定输入。** 记录基模、tokenizer、chat template、停止条件和采样参数；同一段文本经过不同格式化处理，实际训练输入可能不同。
2. **固定轨迹。** 保存生成 token、采样时的 logprob、reward 与 advantage，检查它们是否逐项对应。
3. **检查 loss。** 对齐自回归右移、prompt/tool observation mask、归一化与裁剪。接口或 loss 名称相同，不足以证明计算完全相同。
4. **检查调用完成的时机。** 按各自 SDK 核对 future 与异步返回值，确认参数更新完成后再使用对应权重。
5. **检查保存与评测。** 分清用于推理的权重和可恢复训练的状态，用相同测试集与预算比较结果。checkpoint 路径、登录信息和模型权限需要按各自服务处理。

本仓库的 [GRPO 快速启动](/experiments/grpo/run.html)使用同步入口，可以先从较容易追踪的调用顺序开始；[GRPO、DAPO 与 GSPO 对照](./grpo-dapo-gspo.html)说明不同算法改变了哪些更新环节。

## 按你的目标继续阅读

- **想直接跑 PyTRIO 实验：** 从 [PyTRIO 教程](./pytrio.html)和[安装指南](./quickstart.html)开始。
- **想在 Tinker 上运行训练：** 使用 [Tinker 官方快速开始](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)与 [RL Cookbook](https://tinker-docs.thinkingmachines.ai/cookbook/rl/)，再对照本仓库的算法笔记。
- **想研究 Agent RL：** 阅读 [Search-R1](/experiments/search-r1/)、[ReTool](/experiments/retool/)与 [ALFWorld](/experiments/alfworld/)，先理解环境反馈如何进入训练轨迹。

模型可用性、服务价格和账户权限会变化，请查看各平台当时的官方信息。本页提供工作流阅读与迁移检查方法，不据此判断哪种服务更快或更便宜。
