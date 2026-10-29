# Chapter 0: Reinforcement Learning Loss Functions

![A cartoon explanation of three RL losses](./images/rl_losses_cartoon_cover.png)

<div align="center">
  <a href="https://www.zhihu.com/people/feng-qi-xia-pian" target="_blank"><img alt="Zhihu" src="https://img.shields.io/badge/Zhihu-知乎-4362f6"></a>
  <a href="https://www.xiaohongshu.com/user/profile/63c2055e000000002502c58c" target="_blank"><img alt="Rednote" src="https://img.shields.io/badge/Rednote-小红书-e93c49"></a>
  <a href="https://github.com/datawhalechina/agentic-rl-lab"><img alt="visitors" src="https://komarev.com/ghpvc/?username=KMnO4-zx-agentic-rl-lab-loss-function&amp;label=visitors&amp;color=1283c3&amp;style=flat"></a>
</div>

- Left: Importance Sampling. A model generates tokens, and its choices receive rewards or penalties. The loss increases the probability of useful choices and decreases that of unhelpful ones.
- Middle: PPO. A single reward should not cause an excessive update. Clipping acts as a speed limit on the update.
- Right: CISPO. The clipped ratio becomes a fixed weight that scales the gradient; the gradient flows through the current token's `logprob`.

I recently started studying reinforcement learning with PyTRIO and found myself enjoying the experiments. Previously, the combined training and inference setup in verl, and access to a node with four or eight GPUs, had made it difficult for me to get started. With PyTRIO handling the infrastructure, I can focus on the algorithms. I plan to share reproductions of ten papers, release the code, and cover GRPO, GSPO, DAPO, OPD and its variants.

This is chapter zero of the series: an introduction to reinforcement learning loss functions.

A common first question is: supervised learning penalizes a mismatch with a target answer, but in reinforcement learning the model generates an answer before receiving a reward. What, then, does the loss optimize?

My starting intuition is that many reinforcement learning losses aim to do one thing:

> Make tokens associated with useful outcomes more likely in future, and make tokens associated with unhelpful outcomes less likely.

Three variables connect that intuition to the implementation:

- `advantage`: whether this action was relatively good or bad.
- `sampling_logprobs`: how likely the sampling policy considered the token when it generated the answer.
- `target_logprobs`: how likely the current policy considers that token during training.

These variables help explain the differences between Importance Sampling, PPO and CISPO.

### 1. Importance Sampling

Importance Sampling asks for the ratio between the current policy's probability and the sampling policy's probability for the same token.

```math
\mathcal{L}_{\text{IS}}(\theta) = \mathbb{E}_{x\sim q}\Bigl[\frac{p_\theta(x)}{q(x)}A(x)\Bigr]
```

In code, the probability ratio is `exp(target_logprobs - sampling_logprobs)`. A minus sign turns the objective into a loss for an optimizer that minimizes:

```python
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
loss = -(prob_ratio * advantages).sum()
```

A positive advantage encourages training to increase the token's probability.

A negative advantage encourages training to decrease the token's probability.

The minus sign comes from minimizing the loss. The objective we want to maximize is `ratio * advantage`.

### 2. PPO

PPO addresses another issue: a large ratio can allow an excessively large policy update.

PPO introduces clipping:

```math
\mathcal{L}_{\text{PPO}}(\theta) = -\mathbb{E}_{x \sim q}\left[\min\left(\frac{p_\theta(x)}{q(x)} \cdot A(x), \text{clip}\left(\frac{p_\theta(x)}{q(x)}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}\right) \cdot A(x)\right)\right]
```

Suppose a token has a positive advantage and its ratio has reached `1.8`: the current policy already assigns it substantially more probability than the sampling policy did. PPO limits the positive-advantage surrogate using a bound such as `1.2`, rather than continuing to increase the objective at the `1.8` ratio. The intuition is to learn without taking an excessive step.

### 3. CISPO

CISPO differs from PPO in how it uses clipping.

PPO clips the objective.

CISPO clips the ratio, then treats it as a fixed coefficient multiplying `logprob * advantage`.

```math
\mathcal{L}_{\text{CISPO}}(\theta) = -\mathbb{E}_{x \sim q}\left[\textbf{sg}\left( \text{clip}\left(\frac{p_\theta(x)}{q(x)}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}\right) \right) \cdot \log p_\theta(x) \cdot A(x)\right]
```

The key operation is `detach`. The clipped ratio carries no gradient and acts only as a weight; the gradient instead flows through `target_logprob`.

CISPO thus caps a token's learning weight without directly eliminating its gradient through objective clipping.

### 4. A concise comparison

| Loss | Intuition | Core operation | Common source of confusion |
| ------------------- | -------------------- | --------------------------------------------- | -------------------------------------------------------- |
| Importance Sampling | Reward useful choices | `ratio * advantage` | The loss can be negative because the code minimizes a negative objective. |
| PPO | Put a limit on the update | `min(unclipped, clipped)` | Taking the minimum makes the surrogate conservative. |
| CISPO | Use a fixed learning weight | `detach(clipped_ratio) * logprob * advantage` | After `.detach()`, the ratio carries no gradient; the gradient flows through `logprob`. |

A shorter summary, preserved from the original note:

```text
IS:    有用就提高概率，没用就降低概率；ratio 负责修正采样偏差。
PPO:   还是这么做，但 ratio 不能太离谱，防止策略更新过猛。
CISPO: ratio 也会被 clip，但 clip 后只当固定权重，真正优化 logprob。
```
