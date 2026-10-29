"""定义 ALFWorld 单工具协议和完整多轮工具上下文。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


ALFWORLD_TOOL = {
    "type": "function",
    "function": {
        "name": "alfworld_step",
        "description": (
            "Execute exactly one text action in the current ALFWorld state. "
            "Copy object and receptacle names exactly, including numeric suffixes such as "
            "'apple 1' and 'fridge 1'. Common command forms are: 'go to <receptacle>', "
            "'open <receptacle>', 'close <receptacle>', "
            "'take <object> from <receptacle>', 'move <object> to <receptacle>', "
            "'inventory', 'examine <thing>', 'use <object>', "
            "'heat <object> with <receptacle>', "
            "'clean <object> with <receptacle>', 'cool <object> with <receptacle>', "
            "'slice <object> with <object>', and 'look'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "One ALFWorld command. Use the exact spelling and numbering shown "
                        "by the environment. Do not include explanations in this string."
                    ),
                }
            },
            "required": ["action"],
        },
    },
}

SYSTEM_PROMPT = """You are an autonomous agent in the text-only ALFWorld environment.
Complete the household task using as few valid steps as you can.

At every turn:
1. Reason from the task, complete tool history, current observation, and available actions.
2. Call alfworld_step exactly once with one action string.
3. Wait for the next environment observation before choosing another action.

Object names include instance numbers. For example, "apple 1" and "fridge 1" are complete
environment identifiers; copy them exactly. Never invent abbreviations such as "fridage 1".
Do not claim the task is complete yourself. The environment decides completion."""

PRIVILEGED_SKILL_HEADER = "[Privileged Skill Information]"

XML_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=alfworld_step>\s*"
    r"<parameter=action>\s*(.*?)\s*</parameter>\s*"
    r"</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
GENERIC_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedAction:
    """一次 assistant 输出的工具协议解析结果。"""

    valid_format: bool
    action: str | None
    reasoning: str


def canonical_action(action: str) -> str:
    """匹配官方 projection：转小写并压平多余空白。"""
    return " ".join(action.strip().lower().split())


def _parse_json_tool_call(body: str) -> str | None:
    """兼容部分 tokenizer 输出的 JSON 版 ``<tool_call>``。"""
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or value.get("name") != "alfworld_step":
        return None
    arguments = value.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict) or not isinstance(arguments.get("action"), str):
        return None
    return arguments["action"]


def parse_assistant(text: str) -> ParsedAction:
    """要求恰好一个位于输出末尾的 ``alfworld_step`` 工具调用。"""
    if len(re.findall(r"<tool_call>", text, re.IGNORECASE)) != 1:
        return ParsedAction(False, None, text.strip())
    if len(re.findall(r"</tool_call>", text, re.IGNORECASE)) != 1:
        return ParsedAction(False, None, text.strip())

    xml_matches = list(XML_TOOL_CALL_PATTERN.finditer(text))
    json_matches = list(GENERIC_TOOL_CALL_PATTERN.finditer(text))

    matches: list[tuple[re.Match[str], str]] = []
    matches.extend((match, match.group(1)) for match in xml_matches)
    for match in json_matches:
        # XML 参数体也会被宽松 JSON 正则忽略；这里只接收真正可解析的
        # JSON。
        action = _parse_json_tool_call(match.group(1))
        if action is not None:
            matches.append((match, action))

    if len(matches) != 1:
        return ParsedAction(False, None, text.strip())
    match, raw_action = matches[0]
    if text[match.end() :].strip():
        return ParsedAction(False, None, text.strip())
    action = canonical_action(raw_action)
    if not action or "\n" in raw_action.strip() or "\r" in raw_action.strip():
        return ParsedAction(False, None, text.strip())
    return ParsedAction(True, action, text[: match.start()].strip())


def _available_actions(
    admissible_actions: Sequence[str],
) -> list[str]:
    """规范化环境动作，并排除仅用于调试的 ``help``。"""
    return [
        canonical_action(action)
        for action in admissible_actions
        if action != "help"
    ]


def initial_user_prompt(
    *,
    task: str,
    observation: str,
    admissible_actions: Sequence[str],
    include_admissible_actions: bool,
) -> str:
    """构造一条轨迹的初始 user 消息。"""
    lines = [
        f"Task: {task}",
        "Current step: 1",
        "Current observation:",
        observation,
    ]
    if include_admissible_actions:
        actions = _available_actions(admissible_actions)
        lines.append("Available actions (copy one exactly):")
        lines.extend(f"- {action}" for action in actions)
    lines.append("Choose the next action by calling alfworld_step exactly once.")
    return "\n".join(lines)


def initial_messages(**kwargs: Any) -> list[dict[str, Any]]:
    """返回一条轨迹起点的 system/user 消息。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_user_prompt(**kwargs)},
    ]


def teacher_initial_messages(
    *,
    skill_text: str,
    task: str,
    observation: str,
    admissible_actions: Sequence[str],
    include_admissible_actions: bool,
) -> list[dict[str, Any]]:
    """Return the same initial task with Teacher-only privileged skills.

    The Student never calls this helper. The skill is prepended to the system
    content while the user task, environment observation and tool schema remain
    exactly the same.
    """

    normalized_skill = str(skill_text).strip()
    teacher_system = (
        f"{PRIVILEGED_SKILL_HEADER}\n{normalized_skill}\n\n{SYSTEM_PROMPT}"
    )
    return [
        {"role": "system", "content": teacher_system},
        {
            "role": "user",
            "content": initial_user_prompt(
                task=task,
                observation=observation,
                admissible_actions=admissible_actions,
                include_admissible_actions=include_admissible_actions,
            ),
        },
    ]


def environment_tool_content(
    *,
    step: int,
    observation: str,
    admissible_actions: Sequence[str],
    valid_format: bool,
    admissible: bool,
    done: bool,
    won: bool,
    include_admissible_actions: bool,
) -> str:
    """把一次环境执行结果渲染为下一轮可见的 tool observation。"""
    if won:
        status = "Task completed successfully."
    elif done:
        status = "Episode ended without success."
    elif not valid_format:
        status = "Invalid tool-call format; the environment state did not advance."
    elif not admissible:
        status = "The action was not admissible; choose an available action."
    else:
        status = "Action executed."

    lines = [
        f"ALFWorld step {step} result: {status}",
        "Current observation:",
        observation,
    ]
    if include_admissible_actions and not done:
        lines.append("Available actions (copy one exactly):")
        lines.extend(
            f"- {action}" for action in _available_actions(admissible_actions)
        )
    return "\n".join(lines)


def tool_message(call_id: str, content: str) -> dict[str, Any]:
    """构造一条结构化 ALFWorld tool observation。"""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "alfworld_step",
        "content": content,
    }


def _one_dimensional_tokens(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def _render_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """调用模型原生 chat template，并统一为一维 token。"""
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[ALFWORLD_TOOL],
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    return _one_dimensional_tokens(rendered)


def build_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    """渲染包含完整历史、工具定义和下一轮 assistant 起点的 prompt。"""
    return _render_chat(tokenizer, messages, add_generation_prompt=True)


def _encoded_text_tokens(tokenizer: Any, text: str) -> list[int]:
    """把普通文本编码结果统一为一维 token。"""
    return _one_dimensional_tokens(
        tokenizer.encode(text, add_special_tokens=False)
    )


def _suffix_prefix_overlap(tokens: list[int], suffix: list[int]) -> int:
    """返回采样结尾与 assistant 闭合 token 开头的最长重叠长度。"""
    for length in range(min(len(tokens), len(suffix)), 0, -1):
        if tokens[-length:] == suffix[:length]:
            return length
    return 0


def build_next_prompt(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    next_tool_message: dict[str, Any],
) -> list[int]:
    """在真实采样 token 后追加 assistant 闭合与 tool observation。

    历史 assistant token 始终沿用 sampler 返回值，不把文本重新 tokenize，
    从而保证下一轮 prompt 是上一轮真实 token 序列的严格前缀扩展。
    """
    canonical_prompt = build_prompt(tokenizer, messages_before_assistant)
    placeholder_message = {"role": "assistant", "content": "x"}
    messages_with_assistant = [*messages_before_assistant, placeholder_message]
    canonical_assistant_end = _render_chat(
        tokenizer,
        messages_with_assistant,
        add_generation_prompt=False,
    )
    placeholder_tokens = _encoded_text_tokens(tokenizer, "x")
    canonical_action = [*canonical_prompt, *placeholder_tokens]
    assistant_closing_tokens = canonical_assistant_end[len(canonical_action) :]

    canonical_next_prompt = build_prompt(
        tokenizer,
        [*messages_with_assistant, next_tool_message],
    )
    observation_tokens = canonical_next_prompt[len(canonical_assistant_end) :]

    overlap = _suffix_prefix_overlap(completion_tokens, assistant_closing_tokens)
    return [
        *previous_prompt_tokens,
        *completion_tokens,
        *assistant_closing_tokens[overlap:],
        *observation_tokens,
    ]


def stop_sequences(tokenizer: Any) -> list[str]:
    """返回一轮 assistant 生成使用的停止字符串。"""
    eos_token = getattr(tokenizer, "eos_token", None)
    return [eos_token] if eos_token else []
