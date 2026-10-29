---
title: 参与贡献：一起把 Agentic RL 的门槛降下来
description: Agentic RL Lab 的贡献指南：如何提议新的算法复现章节、章节结构与质量要求、工程规范，以及从 Issue 提案到 PR 合并的完整流程。
---

# 参与贡献

这个仓库的目的只有一个：**让更多的人能以更低的成本，复现 Agentic RL 的算法和论文，真正动手做 Agentic RL 的算法研究**——而不是被没有八卡、被训推一体的复杂度、被 Verl 这类框架的高耦合工程代码挡在门外。

借助 [PyTRIO](https://pytrio.com)，仓库里的每一篇复现都做到了：一杯奶茶的钱、二十个 step、一张消费级显卡都不需要，就能看到真实的训练效果。我们相信，算法研究的价值在于理解算法本身，而不是先被基础设施绊倒三个月。你的每一篇贡献，都是在帮更多和我们一样的人跨过这道门槛。

本页是 [贡献指南](https://github.com/datawhalechina/agentic-rl-lab/blob/main/CONTRIBUTING.md) 的在线版（[English](https://github.com/datawhalechina/agentic-rl-lab/blob/main/CONTRIBUTING_EN.md)）。

## 贡献方式

| 类型 | 流程 |
| --- | --- |
| 新增算法章节 | 先开 Issue 提案 → 维护者确认选题和目录名称 → 提 PR → 审核合并 |
| 修复已有章节的 bug / 补充实验 | 直接提 PR，在描述里说明问题和复现方式 |
| 错别字、文档改进、翻译 | 直接提 PR |

## 新增算法章节的完整流程

1. **开 Issue 提案**。使用[「算法复现提案」模板](https://github.com/datawhalechina/agentic-rl-lab/issues/new/choose)，写清楚：
   - 复现哪篇论文 / 哪个算法（附论文链接）
   - 打算在什么数据集、什么基座模型上复现
   - 训练闭环的 reward / advantage / loss 设计思路
   - 预估的 PyTRIO 成本（参考：现有章节单次训练大多在十几到上百元）
2. **等待确认**。维护者会和你对齐选题是否合适、是否和进行中的工作冲突，并确认 `experiments/<algorithm-name>/` 的目录名称（小写英文与连字符，不加章节编号）。
3. **Fork 并开发**。分支命名建议 `feat/algorithm-name`。
4. **提 PR**。PR 描述里关联 Issue，并附上训练记录和评测结果（见下方质量要求）。
5. **审核**。维护者会从「算法讲得是否明白」和「代码能否复现」两个角度审稿，可能会要求修改几轮，这是正常流程。

## 章节结构约定

新章节请严格对齐现有章节的结构（以 `experiments/search-r1/` 为参考模板）：

```text
experiments/algorithm-name/
├── readme.md          # 章节正文（博客式长文）
├── start.md           # 快速启动：数据准备 → 训练 → 评测的完整命令
├── data.py / prepare_data.py   # 数据下载与预处理
├── reward.py          # reward 计算
├── rollout.py         # 轨迹采样（如适用）
├── train.py           # 训练入口
├── eval.py            # 评测入口
├── analysis.py        # 结果分析与绘图
├── images/            # readme 引用的图片（含封面）
└── .env.example       # 如需 API Key，提供配置模板
```

- 所有脚本都从**仓库根目录**运行，命令形如 `uv run python experiments/algorithm-name/train.py ...`。
- `readme.md` 的风格请参考已有章节：先讲清楚算法从哪来、解决什么问题、核心变量是什么，再讲怎么复现。直觉优先，鼓励用生活化的类比（奶茶、喜茶都可以）。
- `start.md` 必须让一个新读者按顺序复制命令就能跑通：登录（`trio login`、`swanlab login`）→ 数据 → 训练 → 评测。

## 质量要求（合并门槛）

1. **真实跑通**：代码必须在 PyTRIO 上完整跑通过端到端训练，不是只验证 import 或编译。PR 里请附 SwanLab 训练记录链接。
2. **有效果证据**：在固定评测集上给出 Base Model vs 训练后 checkpoint 的对比数字（例如 EM / Format Rate），并说明评测设置。
3. **成本透明**：写明单次训练的实际花费（元）和耗时，方便读者判断是否复现得起。
4. **低门槛优先**：优先选择小模型、小步数就能看到指标变化的配置。本仓库的宗旨是让学习者跑 20 个 step 就能确认「代码是有效的」。
5. **诚实边界**：单次小规模训练能支持什么结论、不能支持什么结论，请在文中写清楚。

## 工程规范

- 依赖用 [uv](https://docs.astral.sh/uv/) 管理，Python 要求 `>=3.13`；新增依赖请先确认现有 `pyproject.toml` 里没有等价物。
- **不要提交**：`.env` 及任何密钥、数据集文件、`swanlog/`、评测输出、checkpoint。数据集和评测输出目录请在 `.gitignore` 中追加对应条目；需要的密钥放进 `.env.example` 模板并留空值。
- 训练实验追踪统一用 [SwanLab](https://swanlab.cn/)。
- 图片统一放在章节目录的 `images/` 下，正文用相对路径引用。

## 语言

章节正文以中文为主（与现有章节一致）；如果你的英文版本更方便，欢迎同时提供。代码注释、变量命名用英文。

## 交流

- 微信交流群：见 [Issue #13](https://github.com/datawhalechina/agentic-rl-lab/issues/13)
- 不确定选题是否合适？直接开 Issue 问，或者在群里聊，不用等有完整方案再开口。

期待你的贡献，一起把 Agentic RL 的门槛降下来！
