"""读取 prepare_data.py 生成的本地 Tournament-GRPO 数据。"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResearchExample:
    """保存一道开放式研究题及作者提供的评分 rubric。"""

    id: str
    query: str
    rubrics: list[dict[str, str]]
    data_source: str


def load_examples(path: str | Path) -> list[ResearchExample]:
    """按文件原顺序读取全部样本；作者训练使用 shuffle=False。"""
    with Path(path).open(encoding="utf-8") as file:
        return [ResearchExample(**json.loads(line)) for line in file]
