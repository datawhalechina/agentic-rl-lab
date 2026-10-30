"""Compare paired ALFWorld evaluations and report actual training budgets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiment import BASE_DIR, write_json

EVAL_KEYS = (
    "base_model", "split", "limit_per_split", "games_per_batch", "max_episode_steps",
    "max_trajectory_tokens", "max_assistant_tokens", "temperature", "top_p", "seed",
    "include_admissible_actions", "environment_asynchronous",
)
TRAIN_KEYS = (
    "base_model", "lora_rank", "train_split", "max_train_games", "max_steps",
    "games_per_batch", "group_size", "max_episode_steps", "max_trajectory_tokens",
    "max_assistant_tokens", "temperature", "top_p", "seed", "learning_rate", "beta1",
    "beta2", "weight_decay", "grad_clip_norm", "ppo_clip_low", "ppo_clip_high",
    "include_admissible_actions", "environment_asynchronous",
)


def named_paths(items: list[str]) -> dict[str, Path]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Use LABEL=PATH")
        label, path = item.split("=", 1)
        if not label or label in result:
            raise ValueError(f"Empty or duplicate label: {label}")
        result[label] = Path(path)
    return result


def load_evaluation(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    games = {}
    summary = None
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if summary is not None:
                raise ValueError(f"Records after final summary in {path}")
            if row.get("type") == "trajectory":
                if row["game_id"] in games:
                    raise ValueError(f"Duplicate game in {path}: {row['game_id']}")
                if row.get("split") not in {"valid_seen", "valid_unseen"}:
                    raise ValueError("Evaluation must contain validation games only")
                if not isinstance(row.get("won"), bool):
                    raise ValueError("won must be boolean")
                games[row["game_id"]] = row
            elif row.get("type") == "summary":
                summary = row
            else:
                raise ValueError(f"Unknown record type in {path}")
    if not games or summary is None:
        raise ValueError(f"Incomplete evaluation (missing trajectories/final summary): {path}")
    for split in ("valid_seen", "valid_unseen"):
        count = sum(row["split"] == split for row in games.values())
        if count and summary["metrics"].get(f"eval/{split}/games") != count:
            raise ValueError(f"Summary count mismatch: {path}, {split}")
    return games, summary


def validate_pair(left: tuple, right: tuple, *, allow_legacy: bool = False) -> bool:
    left_games, left_summary = left
    right_games, right_summary = right
    if set(left_games) != set(right_games):
        raise ValueError("Evaluation game ID sets differ; do not compare partial/different splits")
    if left_summary.get("base_model") != right_summary.get("base_model"):
        raise ValueError("Base models differ")
    for game_id in left_games:
        for key in ("split", "task_type", "task", "initial_observation"):
            if left_games[game_id].get(key) != right_games[game_id].get(key):
                raise ValueError(f"Game definition mismatch: {game_id}, {key}")
    lhs, rhs = left_summary.get("evaluation"), right_summary.get("evaluation")
    if lhs is None or rhs is None:
        if not allow_legacy:
            raise ValueError("Missing evaluation metadata; rerun with this eval.py or explicitly use --allow-legacy-metadata")
        return False
    for key in EVAL_KEYS:
        if lhs["config"].get(key) != rhs["config"].get(key):
            raise ValueError(f"Evaluation setting mismatch: {key}")
    if lhs["dataset_fingerprints"] != rhs["dataset_fingerprints"]:
        raise ValueError("Dataset content fingerprints differ")
    for filename in ("data.py", "eval.py", "protocol.py", "environment.py", "reward.py", "rollout.py"):
        if lhs["provenance"]["source_sha256"][filename] != rhs["provenance"]["source_sha256"][filename]:
            raise ValueError(f"Evaluation implementation mismatch: {filename}")
    if lhs["provenance"]["versions"] != rhs["provenance"]["versions"]:
        raise ValueError("Evaluation package versions differ")
    return True


def aggregate(label: str, split: str, records: list[dict]) -> dict[str, Any]:
    count = len(records)
    steps = [step for row in records for step in row["steps"]]
    return {
        "label": label, "split": split, "games": count,
        "wins": sum(row["won"] for row in records),
        "success_rate": sum(row["won"] for row in records) / count,
        "mean_reward": sum(row["reward"] for row in records) / count,
        "mean_steps": len(steps) / count,
        "mean_invalid_actions": sum(row["invalid_action_count"] for row in records) / count,
        "format_rate": sum(step["valid_format"] for step in steps) / max(len(steps), 1),
        "truncated_rate": sum(row["truncated"] for row in records) / count,
    }


def paired_delta(reference: dict, treatment: dict, ids: list[str], repetitions: int) -> dict:
    import numpy as np
    differences = np.asarray([float(treatment[key]["won"]) - float(reference[key]["won"]) for key in ids])
    rng = np.random.default_rng(42)
    means = rng.choice(differences, size=(repetitions, len(ids)), replace=True).mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "games": len(ids), "delta_percentage_points": float(differences.mean() * 100),
        "paired_bootstrap_95ci_pp": [float(lower * 100), float(upper * 100)],
        "new_successes": int((differences > 0).sum()), "lost_successes": int((differences < 0).sum()),
    }


def training_budgets(paths: dict[str, Path], evaluations: dict[str, tuple] | None = None) -> dict:
    result = {}
    reference = None
    reference_step = None
    for label, path in paths.items():
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("type") != "training_run" or run["status"] != "completed":
            raise ValueError(f"Training did not complete: {path}")
        if run["completed_updates"] != run["config"]["max_steps"]:
            raise ValueError("Completed iteration count does not match the planned budget")
        if reference is not None:
            for key in TRAIN_KEYS:
                if run["config"][key] != reference["config"][key]:
                    raise ValueError(f"Training budget/config mismatch: {key}")
            if run["dataset_fingerprint"] != reference["dataset_fingerprint"]:
                raise ValueError("Training game content/order differs")
            for filename in ("data.py", "train.py", "advantages.py", "reward.py", "protocol.py", "environment.py", "rollout.py"):
                if run["provenance"]["source_sha256"][filename] != reference["provenance"]["source_sha256"][filename]:
                    raise ValueError(f"Training implementation mismatch: {filename}")
            if run["provenance"]["versions"] != reference["provenance"]["versions"]:
                raise ValueError("Training package versions differ")
        reference = run
        checkpoint_step = run["completed_updates"]
        if evaluations is not None:
            if label not in evaluations:
                raise ValueError(f"Training run label has no evaluation: {label}")
            summary = evaluations[label][1]
            if summary["base_model"] != run["config"]["base_model"]:
                raise ValueError(f"Training and evaluation base models differ: {label}")
            matches = {item["update"] for item in run["checkpoints"] if item["sampler_weights"] == summary.get("model_path")}
            if len(matches) != 1:
                raise ValueError(f"Evaluation weights do not identify one checkpoint in this run: {label}")
            checkpoint_step = matches.pop()
            requested = summary.get("evaluation", {}).get("config", {}).get("checkpoint_step")
            if requested is not None and requested != checkpoint_step:
                raise ValueError("Evaluation checkpoint metadata is inconsistent")
        if reference_step is not None and checkpoint_step != reference_step:
            raise ValueError("Evaluated checkpoints use different iteration budgets")
        reference_step = checkpoint_step
        with path.with_name("metrics.jsonl").open(encoding="utf-8") as file:
            metrics = [json.loads(line) for line in file]
        if [row["update"] for row in metrics] != list(range(1, run["completed_updates"] + 1)):
            raise ValueError("Training metric sequence is incomplete")
        metrics = [row for row in metrics if row["update"] <= checkpoint_step]
        if not metrics:
            raise ValueError("Checkpoint has no completed training iterations")
        result[label] = {
            "algorithm": run["config"]["algorithm"], "iterations": len(metrics),
            "optimizer_updates": metrics[-1]["train/optimizer_updates"],
            "rollout_trajectories": sum(row["train/logical_trajectories"] for row in metrics),
            "environment_steps": sum(row["train/logical_steps"] for row in metrics),
            "prefill_tokens_logical": sum(row["budget/prefill_tokens"] for row in metrics),
            "sampled_tokens": sum(row["budget/sampled_tokens"] for row in metrics),
            "training_tokens": sum(row["train/tokens"] for row in metrics),
            "iteration_seconds": sum(row["time/update_seconds"] for row in metrics),
            "actual_cost_yuan": None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="LABEL=eval.jsonl")
    parser.add_argument("--reference", default="GRPO")
    parser.add_argument("--runs", nargs="*", default=[], help="Optional LABEL=outputs/run.json")
    parser.add_argument("--allow-legacy-metadata", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs" / "analysis")
    parser.add_argument("--figure", type=Path, default=BASE_DIR / "images" / "evaluation.png")
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")
    paths = named_paths(args.inputs)
    results = {label: load_evaluation(path) for label, path in paths.items()}
    if args.reference not in results:
        parser.error("reference must be one of the input labels")
    baseline = results[args.reference]
    verified = all([validate_pair(baseline, value, allow_legacy=args.allow_legacy_metadata) for value in results.values()])
    rows = []
    comparisons = {}
    for label, (games, _) in results.items():
        comparisons[label] = {}
        for split in ("all", "valid_seen", "valid_unseen"):
            ids = sorted(key for key, value in games.items() if split == "all" or value["split"] == split)
            if not ids:
                continue
            records = [games[key] for key in ids]
            rows.append(aggregate(label, split, records))
            comparisons[label][split] = paired_delta(baseline[0], games, ids, args.bootstrap_samples)
        for task in sorted({row["task_type"] for row in games.values()}):
            rows.append(aggregate(label, f"task/{task}", [row for row in games.values() if row["task_type"] == task]))
    report = {
        "reference": args.reference, "metadata_verified": verified,
        "inputs": {label: str(path) for label, path in paths.items()},
        "metrics": rows, "paired_comparisons": comparisons,
        "training_budgets": training_budgets(named_paths(args.runs), results),
        "limitations": "Bootstrap intervals reflect resampling these games, not variability across training seeds. Prefill counts are logical inputs, not a billing estimate. Actual cost requires PyTRIO billing records.",
    }
    write_json(args.output_dir / "comparison.json", report)
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    splits = [split for split in ("valid_seen", "valid_unseen", "all") if any(row["split"] == split for row in rows)]
    width = 0.8 / len(results)
    for index, label in enumerate(results):
        values = [next(row["success_rate"] * 100 for row in rows if row["label"] == label and row["split"] == split) for split in splits]
        bars = ax.bar(np.arange(len(splits)) - 0.4 + width / 2 + index * width, values, width, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set(xticks=np.arange(len(splits)), xticklabels=splits, ylabel="Success rate (%)", ylim=(0, 110), title="ALFWorld: identical evaluation games and protocol")
    ax.legend()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    for row in rows:
        if row["split"] in {"all", "valid_seen", "valid_unseen"}:
            print(f"{row['label']:12s} {row['split']:14s} {row['wins']}/{row['games']} = {row['success_rate']:.2%}")
    print(f"Saved comparison to {args.output_dir}; figure: {args.figure}")


if __name__ == "__main__":
    main()
