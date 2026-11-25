import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from eval import evaluate, parse_args


def sample_response(*texts: str) -> SimpleNamespace:
    sequences = [
        SimpleNamespace(tokens=[1], logprobs=[0.0], text=text) for text in texts
    ]
    return SimpleNamespace(sequences=sequences)


def run_evaluation(output: Path, *, token_budget: int, chunk_size: int = 2) -> None:
    rows = [
        {"id": 101, "problem": "1+1", "answer": "2"},
        {"id": 102, "problem": "1+2+3", "answer": "6"},
    ]
    # One token per question byte puts the second prompt exactly at a budget of five.
    sampler = Mock()
    sampler.get_tokenizer.return_value = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: list(messages[-1]["content"].encode())
    )
    sampler.sample_async = AsyncMock(
        side_effect=[
            sample_response(r"\boxed{2}", r"\boxed{2}"),
            sample_response(r"\boxed{6}", "7"),
        ]
    )
    argv = [
        "eval.py", "--val-n", "2", "--chunk-size", str(chunk_size),
        "--max-trajectory-tokens", str(token_budget), "--output", str(output),
    ]
    with (
        patch("sys.argv", argv),
        patch("eval.load_aime25", return_value=rows),
        patch("eval.trio.ServiceClient") as service,
    ):
        service.return_value.create_sampling_client_async = AsyncMock(return_value=sampler)
        asyncio.run(evaluate(parse_args()))


def read_records(output: Path) -> list[dict]:
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]


class ReToolEvaluationTests(unittest.TestCase):
    def test_missing_samples_preserve_only_completed_chunks(self) -> None:
        for chunk_size, expected in [(2, []), (1, [("problem", 101)])]:
            with self.subTest(chunk_size=chunk_size), TemporaryDirectory() as directory:
                output = Path(directory) / "eval.jsonl"
                with self.assertRaisesRegex(
                    RuntimeError, r"102.*请求 2 条轨迹.*实际返回 0 条"
                ):
                    run_evaluation(output, token_budget=5, chunk_size=chunk_size)
                records = read_records(output) if output.exists() else []
                self.assertEqual(
                    [(row["type"], row.get("problem_id")) for row in records], expected
                )

    def test_complete_samples_include_incorrect_unformatted_answer(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "eval.jsonl"
            run_evaluation(output, token_budget=6)
            first, second, summary = read_records(output)
        self.assertEqual([first["num_correct"], second["num_correct"]], [2, 1])
        self.assertFalse(second["generations"][1]["formatted"])
        self.assertEqual(summary["type"], "summary")
        self.assertEqual(summary["problems"], 2)
        self.assertEqual(summary["generations"], 4)
        self.assertEqual(summary["average_at_n"], 0.75)
        self.assertEqual(summary["pass_at_n"], 1.0)


if __name__ == "__main__":
    unittest.main()
