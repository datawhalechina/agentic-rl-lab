"""GSPO 数学正确性奖励：正确为 1、错误为 0，不含任何长度惩罚。

GSPO 论文（arXiv:2507.18071）把奖励一般地写成 verifier reward
r(x, y) ∈ [0, 1]，没有规定本目录使用的具体数学答案检查器。本实现用
math_verify 判断答案等价性，并直接映射为 1/0。
"""

from __future__ import annotations
from dataclasses import dataclass
from math_verify import parse, verify


@dataclass(frozen=True)
class RewardResult:
    """一条 completion 的可解释奖励。

    ``shaped_reward`` 当前恒等于 ``base_reward``，保留该字段是为了
    后续叠加格式/长度 shaping 时不必改动下游接口。
    """

    base_reward: float
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


def score_answer(
    text: str,
    reference: str,
    *,
    completion_tokens: int,
) -> RewardResult:
    """计算 0/1 正确性奖励：答案等价得 1.0，否则 0.0。"""
    answer = extract_last_boxed(text)
    correct = (
        answer is not None
        and answers_equivalent(answer.strip(), reference.strip())
    )
    base_reward = 1.0 if correct else 0.0
    return RewardResult(
        base_reward=base_reward,
        shaped_reward=base_reward,
        correct=correct,
        valid_format=answer is not None,
        answer=answer,
        completion_tokens=completion_tokens,
    )
