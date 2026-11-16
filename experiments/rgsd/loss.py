"""RGSD 的 sampled-token JSD 估计量。

论文的目标（公式 3）是每个位置上 Student 与 rubric-conditioned Teacher 之间的
generalized JSD，在 Teacher 的 top-128 词表上精确求和：

    m     = β·π_T + (1-β)·π_S
    JSD_β = β·KL(π_T ‖ m) + (1-β)·KL(π_S ‖ m)

PyTRIO 拿不到全词表分布（top-k 上限 20，训练侧 target 只能一维），所以这里改用
梯度等价的单样本估计。对 Student 参数 θ 求导，β·KL(π_T‖m) 那一项的梯度恰好和
另一项合并，得到：

    ∇θ JSD_β = (1-β) · E_{y~π_S}[ log(π_S(y)/m(y)) · ∇θ log π_S(y) ]

PyTRIO 的 ``importance_sampling`` 损失是 ``-Σ ratio·A``，在 ratio=1 处的梯度为
``-Σ A·∇log π``。令

    A(y) = -(1-β)·log(π_S/m) = (1-β)·log(β·π_T(y)/π_S(y) + (1-β))

即可在只知道采样 token 两个 logprob 的情况下得到 JSD_β 的无偏梯度估计。

几个性质：
- A 有下界 ``(1-β)·log(1-β)``（Teacher 认为该 token 概率为 0 时），上界随
  ``π_T/π_S`` 对数增长；比 reverse KL 的 ``log π_T - log π_S`` 更不容易被个别
  token 主导。
- β 只能取开区间 (0, 1)：按上面的定义 β=0 或 1 时散度恒为 0。论文全程用 β=0.5，
  此时两种常见的 β 约定完全一致。
- 估计要求 Student 按自身分布无截断采样：temperature=1、top_p=1、top_k=-1。

直接运行本文件会用 torch 在随机分布上核对上述梯度恒等式：

    uv run python experiments/rgsd/loss.py
"""

from __future__ import annotations

import numpy as np


def jsd_token_advantages(
    student_logprobs: np.ndarray, teacher_logprobs: np.ndarray, beta: float
) -> np.ndarray:
    """逐 token 的 JSD_β advantage：``(1-β)·log(β·exp(lp_T - lp_S) + (1-β))``。"""
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    log_ratio = np.asarray(teacher_logprobs, dtype=np.float64) - np.asarray(
        student_logprobs, dtype=np.float64
    )
    # logaddexp 避免 exp(log_ratio) 在 Teacher 远比 Student 自信时溢出。
    log_mixture_over_student = np.logaddexp(np.log(beta) + log_ratio, np.log1p(-beta))
    return (1.0 - beta) * log_mixture_over_student


def sequence_advantages(
    student_logprobs: np.ndarray,
    teacher_logprobs: np.ndarray,
    train_mask: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """一条回答的训练 advantage 与原始逐 token advantage。

    返回 ``(advantages, raw)``：
    - ``raw`` 是逐 token 的 JSD advantage，只用于记录指标；
    - ``advantages`` 已乘上 mask，并除以被训练的 token 数，对应论文公式 3 里的
      ``1/T Σ_t``。batch 层面的缩放见 ``train.py`` 的 ``batch_scale``。
    """
    raw = jsd_token_advantages(student_logprobs, teacher_logprobs, beta)
    mask = np.asarray(train_mask, dtype=np.float64)
    if raw.shape != mask.shape:
        raise ValueError("logprobs 与 train_mask 长度不一致")
    trained = mask.sum()
    if trained == 0:
        return np.zeros_like(raw), raw
    return raw * mask / trained, raw


def _check_gradient_identity() -> None:
    """用 torch 精确核对：估计量的期望梯度 == JSD_β 的真实梯度。"""
    import torch

    torch.manual_seed(0)
    vocab = 64
    for beta in (0.5, 0.3, 0.8):
        logits = torch.randn(vocab, dtype=torch.float64, requires_grad=True)
        teacher = torch.softmax(torch.randn(vocab, dtype=torch.float64) * 2, -1)
        student = torch.softmax(logits, -1)
        mixture = beta * teacher + (1 - beta) * student
        jsd = beta * (teacher * (teacher / mixture).log()).sum() + (1 - beta) * (
            student * (student / mixture).log()
        ).sum()
        (exact,) = torch.autograd.grad(jsd, logits)

        # 对整个词表按 π_S 加权求期望，等价于无限次采样的平均。
        advantages = torch.from_numpy(
            jsd_token_advantages(
                student.detach().log().numpy(), teacher.log().numpy(), beta
            )
        )
        # importance_sampling 在 ratio=1 处的损失梯度是 -Σ A·∇log π。
        surrogate = -(student.detach() * advantages * torch.log_softmax(logits, -1)).sum()
        (estimated,) = torch.autograd.grad(surrogate, logits)
        error = (exact - estimated).abs().max().item()
        assert error < 1e-12, f"beta={beta}: gradient mismatch {error}"
        print(f"beta={beta}: max |exact - estimated| = {error:.1e}")

    # 下界检查：Teacher 概率趋于 0 时 advantage 收敛到 (1-β)·log(1-β)。
    floor = jsd_token_advantages(np.array([0.0]), np.array([-1e9]), 0.5)[0]
    assert abs(floor - 0.5 * np.log(0.5)) < 1e-12
    print("gradient identity check passed")


if __name__ == "__main__":
    _check_gradient_identity()
