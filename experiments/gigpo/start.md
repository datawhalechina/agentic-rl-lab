# GiGPO × ALFWorld：与现有 GRPO 的受控对照

本章在 `experiments/alfworld` 的 PyTRIO 实现上移植 GiGPO 的两层优势估计。这里提供实验设计和运行入口，**尚无真实训练成绩、耗时或账单**；算法原理见 [章节正文](./readme.md)。

本章目录为 `experiments/gigpo/`，不加章节编号。所有命令都在 **agentic-rl-lab 仓库根目录**执行，使用 Bash/Zsh；本地环境支持 macOS/Linux，Windows 请使用 WSL2/Linux 环境。

## 1. 对照什么，保持什么不变

主实验运行两条独立训练线，从同一基座和相同初始化 seed 开始：

| 项目 | GRPO 对照 | GiGPO 实验 |
| --- | --- | --- |
| 入口 | `train.py --algorithm grpo` | `train.py --algorithm gigpo` |
| 基座 / LoRA | Qwen/Qwen3.5-4B / rank 32 | 相同 |
| 数据 | 原章节同一份 train，3,553 个游戏 | 相同文件、筛选和游戏顺序 |
| 训练预算 | 80 次外层迭代 | 80 次外层迭代 |
| 每次迭代 | 8 个游戏 × 每游戏 8 条轨迹 | 相同 |
| 轨迹数预算 | 5,120 条 | 5,120 条 |
| 单轨迹限制 | 50 次环境交互 / 12,000 tokens | 相同 |
| 每轮输出限制 | 2,048 tokens | 相同 |
| 训练采样 | temperature=1.0，top_p=1.0 | 相同 |
| 上下文 / 工具 | 完整历史 / `alfworld_step` | 相同 |
| 奖励 | 成功 1，失败 0，每次非法动作扣 0.1 | 相同 |
| 宏观优势 | 同游戏轨迹奖励减组内均值，不除标准差 | 相同 |
| 微观优势 | 0 | 同观察步骤的折扣回报减组内均值 |
| 优化 | PPO clip 0.8/1.2，Adam lr=1e-6，β=0.9/0.95 | 相同 |
| 附加 rollout / critic / LLM judge | 无 | 无 |

网页正式报告的是 **80 updates**；上游 `train.py` 的参数默认值是 200，因此本章显式设为 80。`max-steps` 是训练迭代数，`max-episode-steps` 才是一次游戏的动作数。

固定迭代数、游戏数与长度上限，不代表两种策略实际消耗相同 token 或执行相同动作数。代码同时记录实际采样 token、逻辑 prefill token、训练 token、交互步数及 optimizer 更新次数。零优势批次会跳过更新但仍计入迭代预算；不要补跑额外迭代来人为凑齐更新次数。

### GiGPO 的具体定义

令 `r[i,t] = -0.1 * invalid[i,t]`，在轨迹最后一步额外加 `1[won]`：

```text
R_i     = sum_t r[i,t]                         # 与原章节总奖励完全一致
A_epi_i = R_i - mean_j(R_j)                    # 每条轨迹等权
G_it    = sum_(k>=t) gamma^(k-t) * r[i,k]       # gamma=0.95
anchor  = (本次游戏组 ID, 动作执行前的原始 observation)
A_step  = G_it - mean(G of all occurrences of this anchor)
A_it    = A_epi_i + step_weight * A_step       # step_weight=1.0
```

默认采用论文中的 **不除标准差变体（GiGPO w/o std）**，以保持原章节 GRPO 的优势尺度。单元素 anchor 组的微观优势为 0。同一游戏组内的所有访问都参与分组，允许跨时间步、也允许同一轨迹重复访问；不同游戏组不混合。使用动作前的观察，不把未来结果、动作内容、时间步编号或完整历史放入 anchor。

每步合成优势只赋给该步真实生成的 assistant tokens；system/user/tool tokens 的优势为 0。模型输入、原始 completion tokens 和 old logprob 保持原章节的严格前缀对齐。即使宏观优势为 0，只要某一步合成优势非零，仍保留该轨迹训练。

默认 anchor 与官方 ALFWorld 实现一样，是原始文字观察的精确匹配，**不是完整的隐藏 PDDL 状态**。部分可观测环境中，相同观察可能对应不同隐藏状态或不同历史；尤其是反复出现的短反馈。主实验保留这一限制并报告匹配率。`--anchor-mode observation_actions` 可将可用动作集合也加入键，用于更保守的补充消融，但不得混作主实验。完整历史本身不作为键，否则通常会使后续步骤几乎无法跨轨迹匹配。

### 与论文/当前官方实现的边界

- 这是针对本仓库的算法移植与受控实验，不能用论文的成功率作为本配置的预期结果。
- 保留 Qwen3.5-4B、LoRA、PyTRIO、完整历史、8×8 轨迹、80 次迭代、无额外 KL 项。论文使用了不同模型、执行框架、上下文结构、奖励尺度及训练预算。
- 宏观优势按每条轨迹等权，保持 `experiments/alfworld` 一致。当前官方代码还支持按步骤统计宏观均值；这里没有引入长度加权。
- 本章按论文折扣回报公式对上述完整 `r[i,t]` 求和，因此未来非法动作惩罚也被折扣。当前官方代码是在计算环境回报后再减当前步骤的非法惩罚；这是为保持本仓库奖励语义而明确选择的差异。
- 同一 raw observation 的策略上下文可能不同，因此这里的微观比较是可观测状态的近似分组，不宣称获得了完整 Markov 状态或原论文的所有理论保证。

## 2. 安装并登录

需要 Python 3.13 或以上，推荐 Python 3.13；远程 PyTRIO 承担模型训练和采样，本机运行文字环境。先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，再执行：

```bash
uv sync --frozen --extra alfworld
export ALFWORLD_DATA="$PWD/experiments/alfworld/datasets/alfworld"
uv run trio login
uv run swanlab login
```

以上 `uv run` 调用的是项目环境中的 `trio login`、`swanlab login`。本实验不需要搜索 API Key，也没有额外 LLM judge。`.env.example` 只提供可选的 SwanLab 环境变量模板；CLI 登录足够。脚本不会自动读取 `.env`，若选择环境变量认证，应在运行进程前导出变量。

## 3. 本地验证与数据准备

先运行不访问远程训练服务的测试：

```bash
uv run --extra alfworld python -m unittest discover -s experiments/gigpo/tests -v
```

测试包括手算回报、GRPO 原函数对照、token/logprob 对齐、零宏观优势仍保留微观信号，以及使用假 sampler/假环境的训练→checkpoint→评测闭环。**离线通过不等于真实 PyTRIO 或真实 ALFWorld 环境已经跑通。**

离线对照测试优先读取同级的 `experiments/alfworld/`；在尚未迁移的旧本地仓库中，兼容读取原 ALFWorld 章节目录。运行和数据准备的默认路径始终使用新版 `experiments/` 布局。

兼容性说明：锁定的 PyTRIO 0.2.9 没有原训练脚本调用的 `Datum.validate_loss_inputs()`。本章在提交前自行检查 PPO 数组的类型、长度与有限值；两种算法共用这项修正。对照测试仅给原函数临时补上同等校验，随后逐项比较构造出来的 Datum，不修改 `experiments/alfworld/`。

首次下载数据：

```bash
uv run --extra alfworld python experiments/gigpo/prepare_data.py --download
```

已有 `experiments/alfworld` 数据时复用它，不需要再次下载：

```bash
uv run --extra alfworld python experiments/gigpo/prepare_data.py
```

默认检查 train=3,553、valid_seen=140、valid_unseen=134，输出 `experiments/gigpo/outputs/data-manifest.json`，其中保存内容指纹和游戏 ID。数量不符时先检查版本和下载完整性；`--allow-different-counts` 仅用于有意采用其他数据版本的实验，不能再声称复用了原固定评测集。

数据默认与 `experiments/alfworld/` 共享。旧本地仓库中的已有数据也可以通过 `--data-root` 指定，不必重新下载；此时将 `ALFWORLD_DATA` 设置为同一数据根目录。

使用其他数据位置时，在准备、训练、评测三类命令中传相同的 `--data-root /absolute/path/to/alfworld`。

## 4. 真实服务的短验证（会计费）

接下来首次调用 PyTRIO。先对两种算法各做 1 次小批次训练，确认环境、模型输出、训练与 checkpoint 链路：

```bash
for algorithm in grpo gigpo; do
  uv run --extra alfworld python experiments/gigpo/train.py \
    --algorithm "$algorithm" \
    --max-steps 1 --games-per-batch 1 --group-size 8 \
    --max-episode-steps 10 --save-every 0 \
    --run-name "alfworld-${algorithm}-smoke" \
    --output-dir "experiments/gigpo/outputs/${algorithm}-smoke" \
    --swanlab-mode online || break
done
```

检查 `metrics.jsonl` 的 `train/update_skipped`。若为 1，该次只验证了采样/保存，没有实际梯度更新；选择新的输出目录、增加少量迭代或换游戏检查，不能据此报告端到端训练成功。短验证 checkpoint 不用于正式对比，正式训练重新从基座初始化。

## 5. 正式的 GRPO 与 GiGPO 成对训练

预算只在这里改一次。默认与原网页相同，若先做较低预算探索，可把 `TRAIN_STEPS=80` 改为 `TRAIN_STEPS=50`；两种算法必须同步改变。

```bash
TRAIN_STEPS=80
EXPERIMENT_SEED=42

for algorithm in grpo gigpo; do
  uv run --extra alfworld python experiments/gigpo/train.py \
    --algorithm "$algorithm" \
    --base-model Qwen/Qwen3.5-4B --lora-rank 32 \
    --train-split train --seed "$EXPERIMENT_SEED" \
    --max-steps "$TRAIN_STEPS" --games-per-batch 8 --group-size 8 \
    --max-episode-steps 50 --max-trajectory-tokens 12000 \
    --max-assistant-tokens 2048 --temperature 1.0 --top-p 1.0 \
    --learning-rate 1e-6 --beta1 0.9 --beta2 0.95 \
    --weight-decay 0.0 --grad-clip-norm 1.0 \
    --ppo-clip-low 0.8 --ppo-clip-high 1.2 \
    --gamma 0.95 --step-weight 1.0 --anchor-mode observation \
    --save-every 20 --checkpoint-steps 50 \
    --run-name "alfworld-${algorithm}-s${EXPERIMENT_SEED}" \
    --output-dir "experiments/gigpo/outputs/${algorithm}-s${EXPERIMENT_SEED}" \
    --swanlab-mode online || break
done
```

`grpo` 模式忽略微观项，按原章节 reward 减均值；同一组原始轨迹的 Datum 与原章节一致，此项由离线测试核对。GRPO 与 GiGPO 各有独立的 LoRA 训练客户端，不会把 GRPO 的最终权重接着用于 GiGPO。

80 次迭代会保存 20/40/50/60/80 和 final 的 state、sampler weights。50 次预算保存 20/40/50 和 final。若某批次没有更新，checkpoint 编号仍表示采样迭代数。

每次运行会在输出目录生成：

```text
run.json       # 参数、数据/代码指纹、版本、完成状态、checkpoint 路径
metrics.jsonl  # 每次迭代的游戏 ID、奖励、覆盖率、时间、token 与更新计数
```

输出目录已经存在时拒绝覆盖，重跑请使用新的 `--output-dir`。当前入口没有实现中断恢复；保存的 state 留作后续恢复开发使用，不能通过重复运行同一命令自动续训。

重点查看 SwanLab 和本地日志：

- `gigpo/nonzero_micro_fraction`：真正获得非零微观优势的步骤比例。
- `gigpo/cross_trajectory_step_fraction`：anchor 跨不同轨迹匹配的覆盖率。
- `gigpo/noninitial_matched_fraction`：排除所有轨迹都相同的首步，后续是否仍有匹配。
- `gigpo/micro_abs_mean`：微观信号的大小。
- reward/success_rate、invalid_actions、平均步数，以及各类 token 和 optimizer 更新次数。

如果除首步外几乎没有匹配、或微观优势一直为零，应如实报告；不要把一个实际退化成 GRPO 的运行称为 GiGPO 已验证有效。

## 6. 在同一个评测入口评测 Base / GRPO / GiGPO

固定 `valid_seen=140`、`valid_unseen=134`，每个游戏一次 rollout，temperature=0.01，评测 seed=42。不要因为训练 seed 改变而同步改变评测 seed。

Base 只需评一次：

```bash
uv run --extra alfworld python experiments/gigpo/eval.py \
  --base-model Qwen/Qwen3.5-4B --split all --games-per-batch 16 \
  --max-episode-steps 50 --max-trajectory-tokens 12000 \
  --max-assistant-tokens 2048 --temperature 0.01 --top-p 1.0 --seed 42 \
  --output experiments/gigpo/eval_results/base.jsonl --swanlab-mode disabled
```

评测两种算法的预算终点。沿用上一节的 `TRAIN_STEPS` 和 `EXPERIMENT_SEED`；若更换终端，先重新设置它们和 `ALFWORLD_DATA`。

```bash
for algorithm in grpo gigpo; do
  uv run --extra alfworld python experiments/gigpo/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --checkpoint-manifest "experiments/gigpo/outputs/${algorithm}-s${EXPERIMENT_SEED}/run.json" \
    --checkpoint-step "$TRAIN_STEPS" \
    --split all --games-per-batch 16 \
    --max-episode-steps 50 --max-trajectory-tokens 12000 \
    --max-assistant-tokens 2048 --temperature 0.01 --top-p 1.0 --seed 42 \
    --output "experiments/gigpo/eval_results/${algorithm}-s${EXPERIMENT_SEED}-step${TRAIN_STEPS}.jsonl" \
    --swanlab-mode disabled || break
done
```

脚本自动从 `run.json` 读取实际 sampler weights 路径，不需要手动替换虚构的 `RUN_ID`。也可以直接传 `--model-path 'trio://实际路径'`，但不能同时传 manifest。输出 JSONL 必须包含末尾 summary 才算完整；中断输出不能拿去算完整成功率。默认拒绝覆盖，必要时显式加 `--overwrite-output`。

若希望与网页相同地观察 Step 40/80，分别对两种算法重复以上评测，修改 `--checkpoint-step` 与输出文件名。不要只给某一种算法挑最好的 checkpoint；主结果使用预先选定的同一终点。

## 7. 配对分析、表格和绘图

```bash
uv run python experiments/gigpo/analysis.py \
  --inputs \
    Base=experiments/gigpo/eval_results/base.jsonl \
    "GRPO=experiments/gigpo/eval_results/grpo-s${EXPERIMENT_SEED}-step${TRAIN_STEPS}.jsonl" \
    "GiGPO=experiments/gigpo/eval_results/gigpo-s${EXPERIMENT_SEED}-step${TRAIN_STEPS}.jsonl" \
  --reference GRPO \
  --runs \
    "GRPO=experiments/gigpo/outputs/grpo-s${EXPERIMENT_SEED}/run.json" \
    "GiGPO=experiments/gigpo/outputs/gigpo-s${EXPERIMENT_SEED}/run.json" \
  --output-dir "experiments/gigpo/outputs/analysis-s${EXPERIMENT_SEED}-step${TRAIN_STEPS}" \
  --figure "experiments/gigpo/images/evaluation-s${EXPERIMENT_SEED}-step${TRAIN_STEPS}.png"
```

分析会检查游戏 ID、数据指纹、评测参数和实现是否一致；也会检查成对训练的预算、模型、seed、实现版本和数据顺序，并确认评测权重属于提供的训练记录、两组 checkpoint 位于同一迭代。如果评测 Step 40，资源计数只累加到 Step 40。输出包括：

- seen / unseen / 全部游戏的成功率，以及六类任务的分项指标；总体按游戏数加权。
- 平均奖励、交互步数、非法动作数、格式正确率、截断率。
- GiGPO 相对 GRPO 的成功率差、新增成功/失去成功数，以及按相同游戏配对 bootstrap 的 95% 区间。
- 实际训练资源计数；真实金额暂为 `null`，需填写实际 PyTRIO 账单，不能把 token 统计冒充账单。
- CSV、JSON 和成功率图。

单 seed 的配对区间只反映这批评测游戏的抽样波动，不能代表训练随机性。预算允许时，用训练 seed 42/43/44 各做一对，分别生成报告后汇总均值与标准差；报告全部 seed，不挑最好的一次。每增加一个 seed 都会增加两次完整训练及对应评测。

历史网页中的 52.92%/56.93% 只作背景，不能替代本次匹配条件下重跑的 GRPO。若必须读取旧 `experiments/alfworld/eval.py` 的 JSONL，可显式加 `--allow-legacy-metadata`；报告会标记元数据未验证，仍要求相同游戏 ID 和初始任务，不能据此声称严格对照。

## 8. 可选消融（主结果完成后再做）

1. `--algorithm gigpo --step-weight 0`：应退化为本章 GRPO；已有离线精确检查，通常不必为此额外付费跑满预算。
2. `--gamma 1.0`：观察微观信号是否依赖折扣带来的效率偏好。
3. `--anchor-mode observation_actions`：测试更保守的可观测状态分组。

每个付费消融都使用新的输出目录，并保留与主实验相同的其余预算。以上不是默认启动的额外训练。

## 9. 文件与来源

`data.py` 复用原筛选与顺序逻辑；`protocol.py`、`environment.py` 保持原实现；`reward.py` 拆出共同奖励；`rollout.py` 增加步骤字段；`advantages.py` 计算两层优势；`train.py` 将每步优势映射到 tokens；`eval.py` 共用评测并记录元数据；`analysis.py` 验证可比性并分析；`experiment.py` 保存指纹与权重清单。

- 原实验：[ALFWorld 章节](https://kmno4-zx.github.io/agentic-rl-lab/experiments/alfworld/)。源代码快照：`bbab8c98761b5e9e5d57f4b4bb899c16dcec0034`。
- 算法：[GiGPO 论文](https://arxiv.org/abs/2505.10978)。
- 官方参考：[core_gigpo.py](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/gigpo/core_gigpo.py)、[ALFWorld anchor](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/agent_system/environments/env_manager.py)、[官方配置](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/examples/gigpo_trainer/run_alfworld.sh)。

提交 PR 前还需要真实服务的端到端记录、固定评测结果、实际成本和耗时。已有 `images/cover.svg` 仅为实验设计封面，不包含虚构结果。
