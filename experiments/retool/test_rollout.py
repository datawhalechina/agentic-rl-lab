"""Unit tests for fit_tool_content.

Run from the repo root:
    uv run python -m unittest discover -s experiments/retool
"""

import unittest
from unittest import mock

import rollout
from rollout import PendingExecution, RolloutConfig, fit_tool_content

MARKER = "[... truncated ...]\n"


class CharTokenizer:
    """One token per character; fails loudly instead of hanging on a runaway loop."""

    def __init__(self, max_calls: int = 1000) -> None:
        self.calls = 0
        self.max_calls = max_calls

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.calls += 1
        if self.calls > self.max_calls:
            raise AssertionError("fit_tool_content did not terminate")
        return [ord(char) for char in text]


def fake_next_prompt(tokenizer, messages, prompt_tokens, completion_tokens, tool_message):
    # Observation costs one token per character of tool output.
    return [*prompt_tokens, *completion_tokens, *([0] * len(tool_message["content"]))]


def pending(prompt_len: int, completion_len: int) -> PendingExecution:
    return PendingExecution(
        trajectory=None,
        code="print(1)",
        call_id="code-0-0-1",
        messages_before_assistant=[],
        assistant_text="",
        prompt_tokens=[1] * prompt_len,
        completion_tokens=[2] * completion_len,
    )


@mock.patch.object(rollout, "build_next_prompt", fake_next_prompt)
class FitToolContentTest(unittest.TestCase):
    def fit(self, content: str, *, prompt_len: int, max_trajectory: int, max_tool: int = 512):
        config = RolloutConfig(
            max_trajectory_tokens=max_trajectory,
            max_tool_response_tokens=max_tool,
        )
        return fit_tool_content(CharTokenizer(), content, pending(prompt_len, 10), config)

    def test_output_that_fits_is_unchanged(self):
        content, next_prompt = self.fit("42\n", prompt_len=100, max_trajectory=8192)
        self.assertEqual(content, "42\n")
        self.assertEqual(len(next_prompt), 100 + 10 + 3)

    def test_long_output_keeps_tail_with_one_marker(self):
        output = "".join(f"line {i}\n" for i in range(400))
        content, _ = self.fit(output, prompt_len=100, max_trajectory=8192, max_tool=200)
        self.assertLessEqual(len(content), 200)
        self.assertTrue(content.startswith(MARKER))
        self.assertEqual(content.count("truncated"), 1)
        self.assertTrue(output.endswith(content[len(MARKER):]))

    def test_long_output_that_never_fits_ends_the_trajectory(self):
        # Only 20 tokens of trajectory budget left: no truncation can fit, so
        # the function must give up instead of looping forever.
        traceback = "Traceback (most recent call last):\n" + "x" * 500 + "\nValueError: bad\n"
        self.assertIsNone(self.fit(traceback, prompt_len=8162, max_trajectory=8192))

    def test_short_output_that_never_fits_ends_the_trajectory(self):
        self.assertIsNone(self.fit("ValueError: bad\n" * 3, prompt_len=8182, max_trajectory=8192))


if __name__ == "__main__":
    unittest.main()
