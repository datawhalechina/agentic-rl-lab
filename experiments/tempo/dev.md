# 9.2 TEMPO 复现笔记

> TEMPO: Test-Time-Scaled Value Estimation with Macro-Step Policy Optimization
> 来源：[Dots 博客](https://studio.dots.ai/dots/tempo-blog.html)（论文尚未发布，超参自定）
>
> 定位：**算法级复现**，只验证机制成立，不追求刷分。9.x 系列（Harness-RL 就绪前的过渡合集）的第二篇，第 9.2 篇。

## 复现方案总览

| 项目 | 方案 |
|---|---|
| 复现目标 | 验证机制：TD 沿 macro-step 回传、critic 估值变准、非终局段产生梯度 |
| 环境数据集 | ALFWorld seen split（复用第 8 篇 `ALFWorldGroup` 的交互与校验；状态持久化层为本篇新增） |
| 算法核心 | macro-step 粒度 GRPO + 生成式 critic（actor 兼任，共享参数） |
| 服务端 loss | pytrio 内置 `importance_sampling`，actor / critic 共用，**不需要自定义 loss** |
| 特权信息 | critic prompt 附带 walkthrough，actor 不可见 |

## 关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| T | 50 轮 | 单局交互上限 |
| H | 10 轮 | macro-step 长度，每局 M = 5 段 |
| N | 4 | 每个保存状态的分支数（actor 的 group 大小） |
| K | 4 | critic 同一状态独立估值次数（critic 的 group 大小） |

## 信号定义

| 信号 | 定义 |
|---|---|
| 环境奖励 | 纯 0/1，成功 = 1（段内几乎全 0，只有成功那一轮为 1）。**不用**第 8 篇的非法动作惩罚，保住 R_max = 1 的前提 |
| critic 输出 | 推理后给出 V̂ ∈ [0, 1]，理解为该状态的期望最终成功率 |
| actor return | 段内环境奖励 + V̂(终点状态) |
| G（TD target） | N 条分支 return 的均值；warm-up 阶段改用完整轨迹的 MC return(0/1) |
| critic reward | −\|V̂ − G\| / R_max；纯 0/1 奖励下剩余回报跨度恒为 1，取 R_max = 1 |
| advantage | actor / critic 都做组内中心化（标准 GRPO） |
| masking | 环境 observation token 的 logprob / advantage 填 0 占位 |
| V̂ 解析 | critic 结尾按严格模板输出数值；解析失败按当组最差 reward 处理 |
| degenerate group | 组内 reward 完全相同时 advantage 全 0，跳过该组（沿用 GRPO 篇惯例） |

## 状态管理

| 操作 | 实现 |
|---|---|
| 保存 | 动作历史 + token 前缀 + 累积奖励（第 8 篇 `EnvironmentState` 只是初始快照，持久化层为本篇新建） |
| 恢复 | TextWorld 确定性：新开环境从初始状态重放动作历史，复用 `reset()` 的同 game file / 初始 observation 校验 |
| 消费 | 每轮训练从已保存的中间状态批续跑 H 轮，终点状态再入库 |
| 成本注意 | 每轮为 N 分支各新开环境重放前缀，game file 加载开销需预估 |

## 训练流程

1. **Warm-up**：离线跑完整轨迹，MC return(0/1) 当 G，先训 critic 几百步。
2. **TD 阶段**：每轮取一批保存状态 → 各采 N 条分支跑 H 轮 → critic 估各终点 → 拼装 return / G → actor 与 critic 的 Datum 一起 `forward_backward` → `optim_step` → 保存权重与新终点。
3. **（可选）前缀 IS 修正**：w = exp(lp_当前(前缀) − lp_旧(前缀)) 乘进 advantage；附录说明前缀 on-policy 时等价，初版省略。

## 验证指标（算法成立的标准）

| 指标 | 期望 |
|---|---|
| critic 估值误差 | 随训练下降 |
| V̂ 与最终 MC return 的相关性 | 随训练上升 |
| 同段内奖励相同的分支 | critic 能拉开 V̂ 差距（博客"放置骑士"的 ALFWorld 版） |
| 非终局 macro-step | actor 也在被更新（TEMPO 核心主张，普通 GRPO 只能干等终局） |
| 对照 sanity | 与第 8 篇完整轨迹 GRPO 比收敛速度即可，不要求赢 |
