"""GSPO 序列级裁剪 loss，对应论文 arXiv:2507.18071 式 (5) 和式 (7)。

每条 completion 的重要性比率为逐 token 概率比的几何平均；整条序列共享
同一个 advantage 和裁剪结果。PyTRIO 提供当前 logprob，sampling logprob 与
advantage 由 rollout 元数据传入。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytrio as trio
import torch

from rollout import RolloutSample


@dataclass(frozen=True)
class GSPOConfig:
    """GSPO 序列级裁剪区间，默认值为论文原文取值。"""

    clip_ratio_low: float = 3e-4
    clip_ratio_high: float = 4e-4

    def __post_init__(self) -> None:
        if self.clip_ratio_low < 0 or self.clip_ratio_high < 0:
            raise ValueError("GSPO clip ratios must be >= 0")


@dataclass(frozen=True)
class GSPOMeta:
    """一条 RolloutSample 的本地 GSPO 元信息。"""

    sampling_logprobs: list[float]
    advantage: float
    completion_tokens: int


def build_datum(
    prompt_tokens: list[int],
    sample: RolloutSample,
) -> tuple[trio.Datum, GSPOMeta]:
    """把 RolloutSample 转成右移对齐的 Datum 与 GSPO 元数据。

    - model_input   = prompt_tokens + completion_tokens[:-1]
    - target_tokens = [0] * (len(prompt_tokens) - 1) + completion_tokens

    prompt 段 target 仅作占位，loss 只读取末尾 completion 区间。
    """
    prompt_tokens = [int(token) for token in prompt_tokens]
    completion_tokens = [int(token) for token in sample.completion_tokens]
    sampling_logprobs = [
        float(value) for value in sample.sampling_logprobs
    ]
    if not prompt_tokens:
        raise ValueError("prompt_tokens must not be empty")
    if not completion_tokens:
        raise ValueError("completion_tokens must not be empty")
    if len(completion_tokens) != len(sampling_logprobs):
        raise ValueError("completion token/logprob lengths must match")

    input_tokens = prompt_tokens + completion_tokens[:-1]
    target_tokens = [0] * (len(prompt_tokens) - 1) + completion_tokens
    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
        },
    )
    meta = GSPOMeta(
        sampling_logprobs=sampling_logprobs,
        advantage=float(sample.advantage),
        completion_tokens=len(completion_tokens),
    )
    return datum, meta


def make_gspo_loss_fn(
    metas: list[GSPOMeta],
    config: GSPOConfig,
    *,
    normalization_sequences: int | None = None,
    normalization_tokens: int | None = None,
) -> Callable[[list[trio.Datum], list[Any]], tuple[Any, dict[str, float]]]:
    """创建 PyTRIO forward_backward_custom 使用的本地 GSPO loss。"""
    if not metas:
        raise ValueError("GSPO loss requires at least one datum")
    if normalization_sequences is None:
        normalization_sequences = len(metas)
    if normalization_sequences < len(metas):
        raise ValueError(
            "normalization_sequences must be >= the number of training sequences"
        )
    train_tokens = sum(meta.completion_tokens for meta in metas)
    if normalization_tokens is None:
        normalization_tokens = train_tokens
    if normalization_tokens < train_tokens:
        raise ValueError(
            "normalization_tokens must be >= the number of training tokens"
        )

    def gspo_loss_fn(
        data: list[trio.Datum],
        logprobs_list: list[Any],
    ) -> tuple[Any, dict[str, float]]:
        if not (len(data) == len(logprobs_list) == len(metas)):
            raise ValueError("GSPO loss got mismatched batch lengths")

        sequence_objectives: list[torch.Tensor] = []
        detached_ratios: list[torch.Tensor] = []
        clip_flags: list[torch.Tensor] = []

        for meta, current_values in zip(metas, logprobs_list, strict=True):
            # 末尾 completion 区间与 sampling_logprobs 一一对应。
            current = current_values.float().reshape(-1)
            if current.numel() < meta.completion_tokens:
                raise ValueError("logprob sequence shorter than completion")
            current_completion = current[-meta.completion_tokens :]
            sampling = torch.as_tensor(
                meta.sampling_logprobs,
                dtype=torch.float32,
                device=current.device,
            )
            if sampling.numel() != meta.completion_tokens:
                raise ValueError(
                    "sampling logprobs must match completion_tokens"
                )

            # 式 (7)：逐 token 概率比的几何平均。
            log_ratio = (current_completion - sampling).mean()
            seq_ratio = torch.exp(log_ratio)

            # 式 (5)：整条序列共享同一个裁剪比率。
            clipped_ratio = torch.clamp(
                seq_ratio,
                min=1.0 - config.clip_ratio_low,
                max=1.0 + config.clip_ratio_high,
            )
            unclipped_objective = seq_ratio * meta.advantage
            clipped_objective = clipped_ratio * meta.advantage
            sequence_objectives.append(
                torch.minimum(unclipped_objective, clipped_objective)
            )

            detached_ratios.append(seq_ratio.detach())
            clip_flags.append(
                (clipped_objective.detach() < unclipped_objective.detach())
                .float()
            )

        # 每条序列等权；退化组的零目标仍包含在原始分母中。
        gspo_loss = (
            -torch.stack(sequence_objectives).sum() / normalization_sequences
        )

        ratios = torch.stack(detached_ratios)
        clips = torch.stack(clip_flags)
        completion_lengths = torch.as_tensor(
            [meta.completion_tokens for meta in metas],
            dtype=clips.dtype,
            device=clips.device,
        )
        metrics = {
            "gspo/loss": float(gspo_loss.detach().item()),
            "gspo/seq_ratio_mean": float(ratios.mean().item()),
            # 退化组计入原始分母，但不计入裁剪分子。
            "gspo/sequence_clip_fraction": float(
                (clips.sum() / normalization_sequences).item()
            ),
            "gspo/token_clip_fraction": float(
                ((clips * completion_lengths).sum() / normalization_tokens).item()
            ),
            "gspo/train_sequences": float(len(metas)),
            "gspo/normalization_sequences": float(normalization_sequences),
            "gspo/active_sequence_fraction": float(
                len(metas) / normalization_sequences
            ),
            "gspo/train_tokens": float(train_tokens),
            "gspo/normalization_tokens": float(normalization_tokens),
            "gspo/active_token_fraction": float(
                train_tokens / normalization_tokens
            ),
        }
        return gspo_loss, metrics

    return gspo_loss_fn
