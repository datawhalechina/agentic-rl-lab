# GiGPO

> 快速启动请查看 [start.md](./start.md)：登录 → 数据准备 → 训练 → 评测。

> **代码与复现资源**
>
> - 本章代码：[训练入口](./train.py) · [优势计算](./advantages.py) · [评测入口](./eval.py) · [结果分析](./analysis.py)
> - GiGPO 论文：[Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978)
> - SwanLab 训练记录：此处待补
>
众所周知，GRPO 作为一个现象级的算法，其标志性的创新点在于采用组内奖励归一化，进而可以省去对价值函数的估计。然而其遗留的一个重大问题是，对结果奖励任务（比如数学推理、多轮搜索等 agent 任务），其 advantage 信号无法做到对回答内 token 的差异化。这也是 agentic RL 中要解决的一个中心问题：信用分配（credit assignment）。对于多轮 agent 任务，GiGPO 是解决这一问题的一次经典尝试，其灵感来自于对不同动作的 Q value 也采用类似组内优势估计的方法。

## 结论放在前面

此处待补

## GiGPO 是什么

GiGPO 对应论文的全称是 Group-in-Group Policy Optimization for LLM Agent Training。从名字可以看出其主要的着眼点是 LLM agent 任务，其特点是过程中与外部环境进行多轮交互。传统的单轮 CoT 推理生成任务并非该工作适合的情况，原因稍后会提到。

我们知道，GRPO 的目标函数（不计 KL regularization）是：

$$
J_{\mathrm{GRPO}}(\theta)
=
\mathbb{E}_{x\sim\mathcal{D},\,\{y_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid x)}
\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}
\min\left(
\rho_{i,t}(\theta)\hat A_{i,t},
\operatorname{clip}\left(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon\right)\hat A_{i,t}
\right)
\right]
$$

其中，

$$
\rho_{i,t}(\theta)
=
\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,<t})}
$$

为新旧策略的概率比。在结果奖励的任务中，同一回答所有 token 的 advantage 是：

$$
\hat A_{i,t}=\hat A_i
=
\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)+\epsilon}
$$

这种对所有动作一视同仁的 advantage 在 agent 任务中有很大的坏处。比如，在 ALFWorld 中的“把杯子洗干净并放到指定桌子上”这个任务里，假设我们获得了两条 rollout：

- **轨迹 A**：找到杯子 → 反复查看无关柜子 → 清洗杯子 → 放到指定桌子上，成功。
- **轨迹 B**：找到杯子 → 清洗杯子 → 放到了错误的位置，失败。

将 GRPO 按完整交互轨迹使用时，轨迹 A 的所有动作共享正优势，因此“反复查看无关柜子”这样的无效操作也能获得正向学习信号。轨迹 B 所有动作共享负优势，因此“找到杯子”“清洗杯子”这些有效操作也受到负向更新信号的影响。这不是我们想看到的情况。

如何解决这一问题呢？注意到 GRPO 版本 advantage 的方向性主要体现的是整条轨迹相对于同组其他轨迹的表现。这一信号在训练中肯定是有价值的，所以应该保留。在此基础上，我们希望进一步区分各个动作的贡献：在相同状态下，这次行动带来的后续回报是否优于其他行动？

传统 actor–critic 方法可以通过学习价值函数来构造这样的局部优势，但需要额外训练 critic。GiGPO 希望保持 GRPO 不依赖 critic 的特点。这里作者有一个巧妙的观察：同一任务采样出的多条轨迹经常经过相同的状态。例如，几次购物尝试都到达同一个商品列表页，或者一次尝试绕路后又回到这个页面。作者把这些轨迹从到达这个页面之后所有的路径和结果收集起来作为一组采样，然后再次使用类似 GRPO 的组内相对估计方法。

具体来说，对具有 $T$ 步的一条轨迹中第 $t$ 步的环境状态 $s$ 和动作 $a$，先计算动作的 return：

$$
G_{i,t}=\sum_{k=t}^{T}\gamma^{k-t}r_{i,k}
$$

然后考虑在同一任务下所有轨迹 $\{J_i\}_{i=1}^{G}$，记录每一次到达 $s$ 是在第几条轨迹的第几步，生成一个集合：

$$
\mathcal I(s)=\{(j,u):s_{j,u}=s\}
$$

计算构造的局部优势：

$$
A_{i,t}^{s}
=
G_{i,t}-\operatorname{mean}_{(j,u)\in\mathcal I(s)}G_{j,u}
$$

将局部优势通过一个不小于 0 的超参数加权之后，再和没有除掉标准差的 $\hat A_{i,t}$（用 $A_{i,t}$ 表示）相加，得到最终可以细化到轨迹中每个环境状态的 advantage：

$$
A_{i,t}^{\mathrm{GiGPO}}=A_{i,t}+\omega A_{i,t}^{s}
$$

论文中主要实验设定 $\omega=1$。在针对 WebShop 的超参敏感性实验中，$\omega=0.8$ 表现最好，0.4–1.2 内相对稳健，过大则会降低性能。此外，论文同时对没有标准化的 $A_{i,t}^{\mathrm{GiGPO}}$ 和分别将两个部分 $A_{i,t}$、$A_{i,t}^{s}$ 标准化之后的 advantage 进行了比较，发现其表现取决于任务集。该设计对过程奖励或结果奖励任务同样适用，只需要改变对 $G_{i,t}$ 的计算。原文主要对 WebShop 和 ALFWorld 任务进行了实验，结果如下。

| 成功率 | GRPO | GiGPO | 提升（百分点） |
| --- | --- | --- | --- |
| ALFWorld | 77.6% | 90.2% | 12.6 |
| WebShop | 66.1% | 75.2% | 9.1 |

从上文我们可以看出来，该方法的好处是，在细化奖励的信用分配时，同时不用更多的数据，复用已经获得的分组采样即可。但与此同时，方法的有效性依赖于如何定义和匹配用于分组的状态。这里需要区分 LLM 的完整决策上下文与 GiGPO 用于分组的 anchor 状态。前者可以包含任务描述、当前观测和历史交互；后者即为当前 agent 所处的环境状态，不要求 LLM 的历史上下文相同。不同轨迹即使经过不同的历史路径，也可能因为到达同一个页面或场景而被归入同一组。对于纯文本的单轮 CoT 生成，如果将状态定义为完整 token 前缀，那么两条推理轨迹一旦分叉，之后通常就无法重新匹配。相比之下，多轮 Agent 任务往往提供明确的外部观测，更容易识别重复出现的中间环境状态，因此更适合发挥 GiGPO 的作用。

不过，相同的当前观测并不一定意味着相同的决策条件，历史中可能包含影响下一步行动的重要信息。如果 anchor 忽略了这些信息，组内比较就可能混合不同的决策情境。GiGPO 的模型输入可以包含历史，但 anchor 分组没有考虑历史差异带来的影响，这是该工作的局限。

## 复现实验设置

此处待补

## 代码实现

此处待补

## 本次实验结果与分析

此处待补

## 训练耗时与成本

此处待补
