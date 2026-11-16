"""RGSD 的 Student / Teacher 输入模板与文本工具。

模板逐字取自论文附录 C.1（Figure 10）：Student 只看问题本身，Teacher 把问题和
rubric 放进同一条 user 消息，并以 transition prompt 结尾。训练脚本和诊断脚本共用
这里的构造函数，保证两处看到的上下文完全一致。
"""

from __future__ import annotations

import re
from typing import Any


TEACHER_RUBRIC_HEADER = "Hidden evaluation criteria that a good response should satisfy:"
TEACHER_TRANSITION = (
    "After understanding these evaluation criteria, provide your own thorough "
    "response to the problem. Address the criteria naturally without explicitly "
    "referencing them."
)

# 论文 B.1 用来统计 rubric 泄漏的短语：出现即视为回答显式提到了评分标准。
RUBRIC_LEAK_PATTERN = re.compile(
    r"criteri(?:on|a)\s*\d+|rubric|evaluation criteria|checklist", re.IGNORECASE
)
THINK_CLOSE = "</think>"


def student_message(question: str) -> str:
    """Student 的 user 消息：只有问题，没有 system prompt 和任何附加指令。"""
    return question.strip()


def teacher_message(question: str, rubrics: list[dict[str, Any]]) -> str:
    """Teacher 的 user 消息：问题 + 编号 rubric + transition prompt。"""
    criteria = "\n".join(
        f"{index}. {item['criterion'].strip()}" for index, item in enumerate(rubrics, 1)
    )
    return (
        f"{question.strip()}\n\n"
        f"{TEACHER_RUBRIC_HEADER}\n"
        f"{criteria}\n\n"
        f"{TEACHER_TRANSITION}"
    )


def render_chat_prompt(
    tokenizer: Any, user_message: str, enable_thinking: bool
) -> list[int]:
    """套用模型原生 chat template，把单条 user 消息编码成 prompt token。"""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    # rendered 已包含模板特殊 token，这里不再重复添加。
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("chat prompt token 为空")
    return prompt_ids


def split_thinking(text: str, thinking_enabled: bool) -> tuple[str, str]:
    """把生成文本拆成 (thinking, answer)。

    开启 thinking 时 chat template 已经把 ``<think>`` 写进 prompt，completion 里只有
    闭合标签，所以按最后一个 ``</think>`` 切分。没有闭合标签时：关闭 thinking 则整段
    都是 answer；开启 thinking 则说明思考被 max_tokens 截断，answer 为空。
    """
    head, separator, tail = text.rpartition(THINK_CLOSE)
    if not separator:
        return (text.strip(), "") if thinking_enabled else ("", text.strip())
    return head.replace("<think>", "").strip(), tail.strip()


def mentions_rubric(text: str) -> bool:
    """文本是否显式提到 rubric / 评分标准。"""
    return RUBRIC_LEAK_PATTERN.search(text) is not None
