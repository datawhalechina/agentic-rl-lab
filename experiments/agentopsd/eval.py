"""在不注入技能的情况下评估 ALFWorld 基座模型或 AgentOPSD checkpoint。

在 experiments/agentopsd 目录运行以下命令评估基座模型：

uv run --extra alfworld python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --split all \
    --games-per-batch 16 \
    --output eval_results/base-qwen35-4b-eval.jsonl \
    --swanlab-mode disabled

使用 train.py 输出的 ``Saved sampler weights`` 路径评估 checkpoint：

uv run --extra alfworld python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path 'trio://RUN_ID/sampler_weights/WEIGHTS_NAME' \
    --split all \
    --games-per-batch 16 \
    --output eval_results/checkpoint-eval.jsonl \
    --swanlab-mode disabled

"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import pytrio as trio
import swanlab
from tqdm import tqdm

from data import default_data_root, discover_games
from rollout import (
    MAX_SEQUENCE_TOKENS,
    RolloutConfig,
    Trajectory,
    rollout_batch,
    trajectory_record,
)


BASE_DIR = Path(__file__).resolve().parent


def _path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析模型、split、输出和 rollout 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
    )
    parser.add_argument(
        "--split",
        choices=["valid_seen", "valid_unseen", "all"],
        default="all",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="每个 split 最多评估多少局；0 表示全部",
    )
    parser.add_argument(
        "--games-per-batch",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
    )
    parser.add_argument(
        "--model-path",
        help="PyTRIO sampler weights 路径；留空即评估 base model",
    )
    parser.add_argument(
        "--max-turns",
        "--max-episode-turns",
        "--max-episode-steps",
        dest="max_turns",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--include-admissible-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--environment-asynchronous",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-trajectory-tokens",
        type=int,
        default=MAX_SEQUENCE_TOKENS,
    )
    parser.add_argument(
        "--max-assistant-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "eval_results" / "eval-results.jsonl",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="显式允许覆盖已有评估 JSONL",
    )
    parser.add_argument(
        "--swanlab-project",
        default="agentic-rl-lab-agentopsd",
    )
    parser.add_argument(
        "--run-name",
        default="agentopsd-alfworld-qwen35-4b-eval",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="disabled",
    )
    args = parser.parse_args(argv)
    args.data_root = _path(args.data_root)
    args.output = _path(args.output)
    return args


def chunks(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def split_metrics(trajectories: list[Trajectory], split: str) -> dict[str, float]:
    """计算一个 split 的成功率和行为指标。"""
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    metrics = {
        f"eval/{split}/reward_mean": mean(
            [trajectory.reward for trajectory in trajectories]
        ),
        f"eval/{split}/success_rate": mean(
            [float(trajectory.won) for trajectory in trajectories]
        ),
        f"eval/{split}/steps_mean": mean(
            [float(len(trajectory.steps)) for trajectory in trajectories]
        ),
        f"eval/{split}/truncated_rate": mean(
            [float(trajectory.truncated) for trajectory in trajectories]
        ),
        f"eval/{split}/valid_tool_call_rate": mean(
            [float(step.valid_format) for step in steps]
        ),
        f"eval/{split}/admissible_action_rate": mean(
            [float(step.admissible) for step in steps]
        ),
        f"eval/{split}/invalid_actions_mean": mean(
            [float(trajectory.invalid_action_count) for trajectory in trajectories]
        ),
        f"eval/{split}/games": float(len(trajectories)),
    }
    by_task: dict[str, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        by_task[trajectory.example.task_type].append(float(trajectory.won))
    for task_type, values in sorted(by_task.items()):
        metrics[f"eval/{split}/success_rate/{task_type}"] = mean(values)
    return metrics


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main(args: argparse.Namespace) -> None:
    """依次评测 seen/unseen，并将逐轨迹结果与 summary 写入一个 JSONL。"""
    selected_splits = (
        ["valid_seen", "valid_unseen"] if args.split == "all" else [args.split]
    )
    games_by_split = {}
    for split in selected_splits:
        games = discover_games(args.data_root, split)
        if args.limit_per_split > 0:
            games = games[: args.limit_per_split]
        games_by_split[split] = games

    args.output.parent.mkdir(parents=True, exist_ok=True)

    service_client = trio.ServiceClient()
    supported_models = [str(model) for model in service_client.get_supported_models()]
    sampling_client = service_client.create_sampling_client(
        base_model=args.base_model,
        model_path=args.model_path,
    )
    tokenizer = sampling_client.get_tokenizer()
    rollout_config = RolloutConfig(
        group_size=1,
        max_turns=args.max_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        include_admissible_actions=args.include_admissible_actions,
        environment_asynchronous=args.environment_asynchronous,
    )

    mode = "w" if args.overwrite_output else "x"
    swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        mode=args.swanlab_mode,
        config=serializable_args(args),
        tags=[
            "alfworld",
            "agentopsd",
            "skill-free",
            "evaluation",
            *selected_splits,
        ],
        job_type="eval",
    )
    all_metrics: dict[str, float] = {}
    try:
        with args.output.open(mode, encoding="utf-8") as file:
            for split in selected_splits:
                split_trajectories: list[Trajectory] = []
                games = games_by_split[split]
                with tqdm(total=len(games), desc=f"Eval {split}", unit="game") as progress:
                    for games_batch in chunks(games, args.games_per_batch):
                        batch = rollout_batch(
                            sampling_client,
                            tokenizer,
                            games_batch,
                            rollout_config,
                            progress_callback=progress.update,
                        )
                        split_trajectories.extend(batch.trajectories)
                        for trajectory in batch.trajectories:
                            record = trajectory_record(trajectory)
                            record["type"] = "trajectory"
                            record["evaluation_split"] = split
                            file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        file.flush()

                metrics = split_metrics(split_trajectories, split)
                all_metrics.update(metrics)

            file.write(
                json.dumps(
                    {
                        "type": "summary",
                        "base_model": args.base_model,
                        "model_path": args.model_path,
                        "skill_free": True,
                        "supported_model_verified": args.base_model in supported_models,
                        "pytrio_version": importlib.metadata.version("pytrio"),
                        "metrics": all_metrics,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        swanlab.log(all_metrics, step=0)
    except KeyboardInterrupt:
        swanlab.finish(state="aborted")
        raise
    except Exception as error:
        swanlab.finish(state="crashed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        swanlab.finish()

    print(json.dumps(all_metrics, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Saved evaluation: {args.output}")


if __name__ == "__main__":
    main(parse_args())
