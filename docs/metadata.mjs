// Page metadata is kept separate from the original research notes.
export const metadata = {
  'tournament-grpo': {
    author: { name: 'MRLE', url: 'https://github.com/mrle0429' },
    seoTitle: 'Tournament-GRPO 原理与复现：研究报告淘汰赛奖励',
    runDescription: '准备研究数据，配置 DeepSeek 搜索、Jina Reader 与裁判，运行 Tournament-GRPO 训练和 Deep Research Bench 评测。',
    en: { subtitle: 'Put research reports through a tournament', seoTitle: 'Tournament-GRPO: Tournament Rewards for Research Reports with PyTRIO', description: 'Compare research reports for the same query in tournaments, turn wins into GRPO rewards, and examine a 20-step PyTRIO run with independent evaluation.', runDescription: 'Prepare research data, configure DeepSeek Search, Jina Reader and the judges, then run Tournament-GRPO training and Deep Research Bench evaluation.', tags: ['Tournament rewards', 'Deep Research'] },
  },
  'loss-functions': {
    seoTitle: '强化学习损失函数：Importance Sampling、PPO 与 CISPO',
    en: { subtitle: 'An intuition for reinforcement learning losses', seoTitle: 'RL Loss Functions: Importance Sampling, PPO and CISPO', description: 'Understand importance sampling, PPO clipping and CISPO through intuitive examples, equations and the policy updates used in this lab.', tags: ['Foundations', 'Objectives'] },
  },
  grpo: {
    seoTitle: 'GRPO 代码实战：用 PyTRIO 在 GSM8K 上训练',
    runDescription: '安装依赖、登录 PyTRIO 与 SwanLab，运行 GSM8K 上的 GRPO 训练，了解采样参数、三种 loss 与日志指标。',
    en: { subtitle: 'Learn from comparisons within a group', seoTitle: 'GRPO with PyTRIO: Code and Training Experiments on GSM8K', description: 'Follow GRPO from group sampling and rewards to policy updates on GSM8K, with runnable PyTRIO code and a comparison of three loss functions.', runDescription: 'Install dependencies, log in to PyTRIO and SwanLab, and launch GRPO on GSM8K with sampling settings, three loss options and training metrics.', tags: ['GSM8K', 'Group-relative advantage'] },
  },
  'general-opd': {
    seoTitle: 'On-Policy Distillation 实战：DeepMath 上的采样、教师打分与 Reverse KL',
    en: { subtitle: 'Teach the student on its own trajectories', seoTitle: 'On-Policy Distillation on DeepMath: Sampling, Teacher Scores and Reverse KL', description: 'Build a minimal on-policy distillation loop on DeepMath-103K: student sampling, teacher scoring and reverse KL, with code and evaluation notes.', tags: ['Distillation', 'DeepMath'] },
  },
  'medical-opd': {
    seoTitle: '医疗 On-Policy Distillation：SAR-OPD、IDT-OPD 与多教师实验',
    runDescription: '准备医疗蒸馏数据与教师模型，运行 Medical SFT、SAR-OPD 和 IDT-OPD，查看配置与实验记录入口。',
    en: { subtitle: 'Balance specialist and general capabilities', seoTitle: 'Medical On-Policy Distillation: SAR-OPD, IDT-OPD and Multiple Teachers', description: 'Start from medical SFT and study SAR-OPD and IDT-OPD, with specialist and general-task evaluations and explicit experimental limitations.', runDescription: 'Prepare medical distillation data and teacher models, then run Medical SFT, SAR-OPD and IDT-OPD using the supplied commands and configuration.', tags: ['Medical tasks', 'Multiple teachers'] },
  },
  'search-r1': {
    seoTitle: 'Search-R1 原理与实现：训练会多轮搜索的 LLM Agent',
    runDescription: '配置 DeepSeek 官方 Search API 密钥或免密钥 Wikipedia，运行 Search-R1 多轮搜索训练与 Base / checkpoint 评测。',
    en: { subtitle: 'Learn to search while reasoning', seoTitle: 'Search-R1 Explained: Training LLM Agents for Multi-turn Search', description: 'Train multi-turn search with the official DeepSeek Search API or Wikipedia, and compare Base with the 30-step checkpoint using fixed evaluation questions.', runDescription: 'Configure an official DeepSeek Search API key or key-free Wikipedia, then run Search-R1 multi-turn training and Base / checkpoint evaluation.', tags: ['Online search', 'Multi-turn interaction'] },
  },
  opsd: {
    seoTitle: 'OPSD 自蒸馏实战：固定教师与学生自采样轨迹',
    runDescription: '运行 OPSD 数学推理自蒸馏，配置固定 step-0 Teacher、采样预算与训练指标记录。',
    en: { subtitle: 'Distill from self-generated trajectories', seoTitle: 'OPSD Self-Distillation: A Fixed Teacher and Student-generated Trajectories', description: 'Study self-distillation with a fixed step-0 teacher and trajectories sampled by the student, including the training loop and mathematical reasoning results.', runDescription: 'Launch OPSD mathematical reasoning experiments with a fixed step-0 teacher, configurable sampling and SwanLab training metrics.', tags: ['Self-distillation', 'Mathematical reasoning'] },
  },
  retool: {
    seoTitle: 'ReTool 原理与代码：Python 工具调用与 Agent 强化学习',
    runDescription: '连接 ReTool 本地 Python 沙箱，启动思考与代码执行交织的训练，检查工具输出与 rollout 指标。',
    en: { subtitle: 'Make Python part of the reasoning process', seoTitle: 'ReTool Explained: Python Tool Use and Agent Reinforcement Learning', description: 'Connect a local Python sandbox and study reinforcement learning with interleaved reasoning and code execution, including rollout handling and results.', runDescription: 'Connect the local Python sandbox and run ReTool training with interleaved reasoning, code execution and rollout metrics.', tags: ['Code sandbox', 'Tool use'] },
  },
  dapo: {
    seoTitle: 'DAPO 原理与训练实战：动态采样、裁剪和四项改进',
    runDescription: '启动 DAPO 训练，配置 Dynamic Sampling 与采样参数，查看四项改进相关的训练日志。',
    en: { subtitle: 'Unpack four changes to training stability', seoTitle: 'DAPO Explained: Dynamic Sampling, Clipping and Training Experiments', description: 'Examine DAPO’s four core changes and the practical time cost of dynamic sampling through training code, equations and recorded experiments.', runDescription: 'Run DAPO training with dynamic sampling and configurable rollout settings, and inspect the metrics associated with its four core changes.', tags: ['Dynamic sampling', 'Training stability'] },
  },
  gspo: {
    seoTitle: 'GSPO 原理与核心 Loss 实现：序列级重要性采样',
    runDescription: '运行 GSPO 核心 loss 实验，设置策略更新与采样参数，核对当前实现和完整训练复现的边界。',
    en: { subtitle: 'Optimize at the sequence level', seoTitle: 'GSPO Explained: Sequence-level Importance Sampling and Core Loss Code', description: 'Implement sequence-level importance ratios and clipping, and distinguish the core GSPO loss reproduced here from a complete training reproduction.', runDescription: 'Run the GSPO core-loss experiment with configurable sampling and policy updates, and check the implementation’s reproduction boundaries.', tags: ['Sequence-level optimization', 'Custom loss'] },
  },
  alfworld: {
    seoTitle: 'ALFWorld Agent 强化学习实战：TextWorld 环境与长轨迹训练',
    runDescription: '安装 ALFWorld 和 TextWorld 依赖、准备环境数据，运行家务 Agent 训练与评测命令。',
    en: { subtitle: 'Train an agent to complete household tasks', seoTitle: 'ALFWorld Agent RL: TextWorld Environments and Long-Trajectory Training', description: 'Train a household agent in a real TextWorld environment, with long trajectories, environment feedback, group-relative advantages and evaluation notes.', runDescription: 'Install ALFWorld and TextWorld dependencies, prepare environment data, and run the household-agent training and evaluation commands.', tags: ['Interactive environments', 'Long trajectories'] },
  },
  'vision-grpo': {
    seoTitle: 'Vision GRPO 多模态强化学习：GeoQA 图文推理实战',
    runDescription: '准备 GeoQA 图片数据，配置多模态输入，运行 Vision GRPO 训练与几何问答评测。',
    en: { subtitle: 'Bring images into reinforcement learning', seoTitle: 'Vision GRPO: Multimodal Reinforcement Learning on GeoQA', description: 'Connect image inputs, geometry reasoning and verifiable rewards on GeoQA, with multimodal training code and evaluation details.', runDescription: 'Prepare GeoQA image data, configure multimodal inputs, and run Vision GRPO training and geometry question-answering evaluation.', tags: ['Vision-language models', 'GeoQA'] },
  },
  tempo: {
    seoTitle: 'TEMPO 算法级复现：Macro-step 优化与生成式 Critic',
    en: { subtitle: 'Learn long tasks through shorter trajectory segments', seoTitle: 'TEMPO Algorithm-level Reproduction: Macro-step Optimization and a Generative Critic', description: 'Explore macro-step optimization and a generative critic in ALFWorld, with clear boundaries between the public TEMPO description and this reproduction.', tags: ['Macro-steps', 'Generative critic'] },
  },
  agentopsd: {
    seoTitle: 'AgentOPSD 实战：Self-Teacher 与多轮 Agent 信用分配',
    runDescription: '准备 ALFWorld 与 Skill，运行 AgentOPSD 训练、checkpoint 评测和配对分析，核对模型与采样预算。',
    en: { subtitle: 'Assign credit to each interaction turn', seoTitle: 'AgentOPSD: Self-Teachers and Credit Assignment for Multi-turn Agents', description: 'Use a skill-guided self-teacher to provide turn-level learning signals, then examine multi-turn agent training and paired evaluation in ALFWorld.', runDescription: 'Prepare ALFWorld and skills, then run AgentOPSD training, checkpoint evaluation and paired analysis with explicit models and sampling budgets.', tags: ['Credit assignment', 'Self-teacher'] },
  },
  'spec-o3': {
    seoTitle: 'Spec-o3 多模态 Agent 实战：SFT、GRPO 与天体光谱审核',
    runDescription: '下载 Spec-o3 数据，运行 SFT、GRPO、开发集评测与绘图，核对 Qwen 模型、checkpoint 和结果口径。',
    en: { subtitle: 'Search the sky with reinforcement learning', seoTitle: 'Spec-o3 Multimodal Agents: SFT, GRPO and Astronomical Spectrum Review', description: 'Move from SFT to GRPO for astronomical candidate review, interleaving spectrum images, reasoning and tools with explicit evaluation settings.', runDescription: 'Download Spec-o3 data and run SFT, GRPO, development-set evaluation and plotting, with explicit Qwen models and checkpoint settings.', tags: ['Spectral analysis', 'Multimodal tools'] },
  },
  'pytrio-jev': {
    seoTitle: 'PyTrio-Jev 决策模型实战：LoRA 训练、分类推理与温度校准',
    runDescription: '准备 Typed Decisions、ToolACE 与 Jevbench 数据，运行 Qwen3.5-4B LoRA 训练、决策推理、温度校准和五项基准评测。',
    en: { subtitle: 'Train your own classification and decision model', seoTitle: 'PyTrio-Jev Decision Models: LoRA Training, Classification and Temperature Calibration', description: 'Fine-tune Qwen3.5-4B on 2,160 records with LoRA, then run restricted-softmax inference, temperature calibration and five benchmark evaluations.', runDescription: 'Prepare Typed Decisions, ToolACE and Jevbench data, then run Qwen3.5-4B LoRA training, decision inference, temperature calibration and five benchmark evaluations.', tags: ['Classification', 'Probability calibration'] },
  },
}

export const englishGroups = {
  foundations: { label: 'Foundations & optimization', note: 'Understand a policy update' },
  distillation: { label: 'Distillation & self-learning', note: 'Turn feedback into a learning signal' },
  agents: { label: 'Tools & environments', note: 'Let a model interact with the world' },
  frontiers: { label: 'Multimodal & frontier work', note: 'Explore longer, more complex tasks' },
}
