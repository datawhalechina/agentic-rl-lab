"""读取并循环消费 ``prepare_data.py`` 生成的 DAPO-Math 数据（experiments/gspo 版）。"""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import random


@dataclass(frozen=True)
class MathExample:
    """一道数学训练题。"""

    id: str
    question: str
    answer: str
    data_source: str


def load_examples(path: str | Path) -> list[MathExample]:
    """读取 JSONL，并对缺字段或空数据快速失败。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"找不到训练数据：{source}\n"
            "请先运行：uv run python experiments/gspo/prepare_data.py"
        )

    examples: list[MathExample] = []
    with source.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                example = MathExample(
                    id=str(row["id"]),
                    question=str(row["question"]).strip(),
                    answer=str(row["answer"]).strip(),
                    data_source=str(row.get("data_source") or "math_dapo"),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"无效数据行：{source}:{line_number}") from error
            if not example.question or not example.answer:
                raise ValueError(f"题目或答案为空：{source}:{line_number}")
            examples.append(example)

    if not examples:
        raise ValueError(f"数据文件为空：{source}")
    return examples


def shuffled_examples(path: str | Path, seed: int) -> list[MathExample]:
    """读取并按固定 seed 打乱。"""
    examples = load_examples(path)
    random.Random(seed).shuffle(examples)
    return examples


class ExampleCursor:
    """可回绕的数据游标，训练循环按 step 顺序取题，越过末尾时循环。"""

    def __init__(self, examples: list[MathExample]) -> None:
        if not examples:
            raise ValueError("ExampleCursor requires at least one example")
        self._examples = examples
        self.consumed = 0

    @property
    def position(self) -> int:
        """下一个样本在打乱后数据中的位置。"""
        return self.consumed % len(self._examples)

    def take(self, count: int) -> list[MathExample]:
        """取 ``count`` 道题并推进游标，越过末尾时循环。"""
        if count < 1:
            raise ValueError("count must be >= 1")
        batch = [
            self._examples[(self.consumed + offset) % len(self._examples)]
            for offset in range(count)
        ]
        self.consumed += count
        return batch
