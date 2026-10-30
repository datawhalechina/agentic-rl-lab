# Adapted from experiments/alfworld/data.py; share the baseline dataset directory.
"""发现并按固定顺序读取 ALFWorld 的 text-only 游戏。

从仓库根目录下载共享数据：

uv run --extra alfworld alfworld-download \
    --data-dir "experiments/alfworld/datasets/alfworld"
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "alfworld" / "datasets" / "alfworld"

SPLIT_DIRECTORIES = {
    "train": "train",
    "valid_seen": "valid_seen",
    "valid_unseen": "valid_unseen",
}

SUPPORTED_TASK_TYPES = frozenset(
    {
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
    }
)


@dataclass(frozen=True)
class GameExample:
    """一局可复现的 ALFWorld 游戏。"""

    id: str
    split: str
    game_file: Path
    task_type: str


def default_data_root() -> Path:
    """返回项目内数据目录；其他位置请通过 ``--data-root`` 显式传入。"""
    return DEFAULT_DATA_ROOT.resolve()


def _dataset_root(data_root: str | Path) -> Path:
    """兼容传入缓存根目录、json_2.1.1 目录或某个 split 目录。"""
    root = Path(data_root).expanduser().resolve()
    if root.name in SPLIT_DIRECTORIES.values():
        return root.parent
    nested = root / "json_2.1.1"
    return nested if nested.is_dir() else root


def split_directory(data_root: str | Path, split: str) -> Path:
    """解析一个公开 split 对应的磁盘目录。"""
    if split not in SPLIT_DIRECTORIES:
        choices = ", ".join(SPLIT_DIRECTORIES)
        raise ValueError(f"未知 split: {split!r}；可选值为 {choices}")
    root = _dataset_root(data_root)
    if root.name == SPLIT_DIRECTORIES[split]:
        return root
    return root / SPLIT_DIRECTORIES[split]


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层应为对象: {path}")
    return value


def discover_games(
    data_root: str | Path,
    split: str,
    *,
    task_types: set[str] | frozenset[str] | None = None,
) -> list[GameExample]:
    """发现指定 split 中所有受支持且标记为 solvable 的游戏。"""
    directory = split_directory(data_root, split)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"找不到 ALFWorld 数据目录: {directory}\n"
            "请按 data.py 文件开头的命令下载数据，或显式传入 --data-root"
        )

    allowed = SUPPORTED_TASK_TYPES if task_types is None else frozenset(task_types)
    examples: list[GameExample] = []
    for game_file in sorted(directory.rglob("game.tw-pddl")):
        path_text = str(game_file)
        if "movable" in path_text or "Sliced" in path_text:
            continue

        trajectory_file = game_file.with_name("traj_data.json")
        if not trajectory_file.is_file():
            continue
        trajectory = _load_json(trajectory_file)
        task_type = str(trajectory.get("task_type", ""))
        if task_type not in allowed:
            continue

        game_data = _load_json(game_file)
        if not bool(game_data.get("solvable", False)):
            continue

        relative = game_file.relative_to(directory)
        examples.append(
            GameExample(
                id=f"{split}:{relative.parent.as_posix()}",
                split=split,
                game_file=game_file.resolve(),
                task_type=task_type,
            )
        )

    if not examples:
        raise ValueError(
            f"{directory} 中没有找到可用的 game.tw-pddl；"
            "请检查 alfworld-download 是否完整"
        )
    return examples


def shuffled_games(
    data_root: str | Path,
    split: str,
    seed: int,
    *,
    max_games: int = 0,
) -> list[GameExample]:
    """发现游戏后按固定种子打乱，并可限制本次使用数量。"""
    games = discover_games(data_root, split)
    random.Random(seed).shuffle(games)
    if max_games > 0:
        games = games[:max_games]
    return games


def take_batch(
    games: list[GameExample],
    start: int,
    batch_size: int,
) -> list[GameExample]:
    """从固定游戏序列循环取一个 batch。"""
    if batch_size < 1:
        raise ValueError("batch_size 必须大于等于 1")
    if not games:
        return []
    return [games[(start + offset) % len(games)] for offset in range(batch_size)]
