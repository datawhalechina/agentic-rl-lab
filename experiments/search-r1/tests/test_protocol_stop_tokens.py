"""回归测试：parse_assistant 必须先剥掉尾部停止标记再解析（#58）。

标签在源码里用拼接构造，避免测试文件本身携带模板特殊标记。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol import parse_assistant  # noqa: E402

L, R, S = "<", ">", "/"

TC = (
    L + "tool_call" + R
    + L + "function=search" + R
    + L + "parameter=query" + R
    + "\nFuzhou weather\n"
    + L + S + "parameter" + R
    + L + S + "function" + R
    + L + S + "tool_call" + R
)
IM_END = "<" + "|" + "im_end" + "|" + ">"
TOOL_RESPONSE_CLOSE = L + S + "tool_response" + R


def test_clean_call_is_tool():
    assert parse_assistant(TC).kind == "tool"


def test_trailing_im_end_is_tool():
    assert parse_assistant(TC + IM_END).kind == "tool"


def test_trailing_im_end_with_newline_is_tool():
    assert parse_assistant(TC + "\n" + IM_END + "\n").kind == "tool"


def test_trailing_tool_response_close_is_tool():
    assert parse_assistant(TC + TOOL_RESPONSE_CLOSE).kind == "tool"


def test_query_is_preserved_after_stripping():
    parsed = parse_assistant(TC + IM_END)
    assert parsed.query == "Fuzhou weather"


def test_real_extra_text_still_invalid():
    assert parse_assistant(TC + " and that is all").kind == "invalid"


def test_double_call_still_invalid():
    assert parse_assistant(TC + TC).kind == "invalid"
