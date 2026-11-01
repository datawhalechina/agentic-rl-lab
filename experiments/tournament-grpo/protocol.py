"""Tournament-GRPO 的研究 prompt 与 XML 工具协议。

Adapted from PuffYang/Tournament_GRPO (eea826cf), Apache-2.0.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# 作者 prompt.txt 原文，按作者代码读取后 strip。
SYSTEM_PROMPT = (Path(__file__).resolve().parent / "prompt.txt").read_text(encoding="utf-8").strip()
MAX_PROMPT_LENGTH = 4096
TOOL_STOP = "</call_tool>"
# 低成本配置的收尾提示，放在最后一次工具 observation 内，不作为模型输出训练。
TOOL_BUDGET_NOTICE = (
    "Environment notice: tool budget exhausted; no tool calls remain. "
    "Do not call google_search or browse_webpage again. "
    "Use the evidence already collected to give your final response now: "
    "a think block followed by one answer block, following the required XML format. "
    "If evidence is incomplete, state the limitations rather than inventing facts."
)
TOOL_CALL_PATTERN = re.compile(r'<call_tool\s+name="([^"]+)"([^>]*)>(.*?)</call_tool>', re.DOTALL)
ATTRIBUTE_PATTERN = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
# 作者工具的 OpenAI schema。rollout 走 XML 协议不传 schema，这里只用于复现作者的数据筛长。
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "General web search for relevant webpages and snippets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_webpage",
            "description": "Open a specific URL and extract readable page text.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The webpage URL to browse."}},
                "required": ["url"],
            },
        },
    },
]


@dataclass(frozen=True)
class ToolCall:
    """一次 XML 工具调用，参数名与作者一致（query / url）。"""

    name: str
    arguments: dict[str, str]


def render_chat(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> list[int]:
    """用 chat template 渲染消息，并追加 assistant 生成前缀。"""
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=True, return_dict=False,
    )


def build_prompt(tokenizer: Any, query: str) -> list[int]:
    """一道研究题的初始 prompt：研究 system prompt + 原 user 问题。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    return render_chat(tokenizer, messages)


def parse_tool_call(text: str) -> ToolCall | None:
    """只执行第一个调用，对应作者的 max_parallel_calls=1。"""
    match = TOOL_CALL_PATTERN.search(text)
    if match is None:
        return None
    name, attributes, body = match.groups()
    name = name.strip()
    arguments = dict(ATTRIBUTE_PATTERN.findall(attributes))
    if body.strip():
        arguments["url" if name == "browse_webpage" else "query"] = body.strip()
    return ToolCall(name, arguments)


def truncate_tool_text(text: str) -> str:
    """作者的按字符中间截断：超过 3000 字符时保留首尾各 1500。"""
    return text if len(text) <= 3000 else text[:1500] + "...(truncated)..." + text[-1500:]


def tool_observation(text: str, *, tools_exhausted: bool = False) -> str:
    """observation 直接接在 assistant 后面；预算用尽时在同一 tool_output 内提示收尾。"""
    text = text.strip()
    if tools_exhausted:
        # 保留一对调用/返回，不增加新的顶层块；提示不用 XML 标签，避免干扰格式奖励。
        if text.endswith("</tool_output>"):
            text = text.removesuffix("</tool_output>").rstrip()
        else:
            text = f"<tool_output>\n{text}"  # 工具失败时返回的是没有标签的错误文本。
        text += f"\n\n{TOOL_BUDGET_NOTICE}\n</tool_output>"
    return f"\n{text}\n"


def transcript_text(tokenizer: Any, prompt_tokens: list[int], response_tokens: list[int]) -> str:
    """还原送去格式检查的完整文本；Qwen3.5 把 <think> 放在 prompt 里，这里补回开头。"""
    prefix = re.search(r"(<think>\s*)$", tokenizer.decode(prompt_tokens))
    return (prefix.group(1) if prefix else "") + tokenizer.decode(response_tokens, skip_special_tokens=True)
