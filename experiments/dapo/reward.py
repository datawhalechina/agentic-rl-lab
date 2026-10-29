"""DAPO 数学正确性奖励与 Soft Overlong Punishment。"""

from __future__ import annotations
from dataclasses import dataclass
from math_verify import parse, verify


ANSWER_WINDOW_CHARS = 300


@dataclass(frozen=True)
class RewardResult:
    """一条 completion 的可解释奖励。"""

    base_reward: float
    length_penalty: float
    shaped_reward: float
    correct: bool
    valid_format: bool
    answer: str | None
    completion_tokens: int


def extract_last_boxed(text: str) -> str | None:
    """从末尾向前找到最后一个完整的 ``\\boxed{...}``，支持嵌套括号。"""
    end = len(text)
    while True:
        marker = text.rfind("\\boxed", 0, end)
        if marker < 0:
            return None
        left = text.find("{", marker)
        if left < 0:
            end = marker
            continue
        depth = 0
        for position in range(left, len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
                if depth == 0:
                    return text[left + 1 : position].strip()
        end = marker


def answers_equivalent(prediction: str, reference: str) -> bool:
    """使用 ``math_verify`` 判断两个数学答案是否等价。"""
    try:
        return bool(verify(parse(f"${reference}$"), parse(f"${prediction}$")))
    except Exception:
        return False


def soft_overlong_penalty(
    completion_tokens: int,
    *,
    max_tokens: int,
    overlong_cache: int,
) -> float:
    """按 DAPO 的分段线性规则返回 ``[-1, 0]`` 长度惩罚。"""
    if completion_tokens < 0:
        raise ValueError("completion_tokens must be >= 0")
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if not 1 <= overlong_cache <= max_tokens:
        raise ValueError("overlong_cache must be in [1, max_tokens]")

    penalty_start = max_tokens - overlong_cache
    if completion_tokens <= penalty_start:
        return 0.0
    if completion_tokens <= max_tokens:
        return -(completion_tokens - penalty_start) / overlong_cache
    return -1.0


def score_answer(
    text: str,
    reference: str,
    *,
    completion_tokens: int,
    max_tokens: int,
    overlong_cache: int,
    enable_overlong: bool,
) -> RewardResult:
    """计算 ±1 正确性奖励，并按 preset 可选叠加长度惩罚。"""
    answer = extract_last_boxed(text[-ANSWER_WINDOW_CHARS:])
    correct = (
        answer is not None
        and answers_equivalent(answer.strip(), reference.strip())
    )
    base_reward = 1.0 if correct else -1.0
    length_penalty = (
        soft_overlong_penalty(
            completion_tokens,
            max_tokens=max_tokens,
            overlong_cache=overlong_cache,
        )
        if enable_overlong
        else 0.0
    )
    return RewardResult(
        base_reward=base_reward,
        length_penalty=length_penalty,
        shaped_reward=base_reward + length_penalty,
        correct=correct,
        valid_format=answer is not None,
        answer=answer,
        completion_tokens=completion_tokens,
    )
