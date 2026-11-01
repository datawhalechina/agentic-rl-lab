"""Deep Research Bench 评测：先生成研究报告，再用 RACE 对照参考报告评分。

uv run python experiments/tournament-grpo/eval.py generate
uv run python experiments/tournament-grpo/eval.py score

两个阶段都逐题写入 jsonl；同一命令重跑时会跳过已完成的题目。
"""

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import pytrio as trio
import swanlab
from tqdm import tqdm

from data import ResearchExample
from protocol import transcript_text
from race import RACE_DIMS, RaceJudge
from reward import analyze_format
from rollout import RolloutConfig, rollout_batch
from search import ResearchTools

BASE_DIR = Path(__file__).resolve().parent
FINISH_REASONS = ("no_tool_call", "max_turns", "max_tool_calls", "max_response_tokens")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(file, record: dict) -> None:
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
    file.flush()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def path_name(value: str) -> str:
    """把模型名或远程权重路径转换成单层目录名。"""
    return re.sub(r"[^A-Za-z0-9._-]+", "--", value).strip("-")


def generation_metrics(records: list[dict]) -> dict[str, float]:
    count = len(records)
    calls = sum(len(r["tool_calls"]) for r in records)
    finishes = Counter(r["finish_reason"] for r in records)
    return {
        "eval/questions": count,
        "eval/answer_rate": sum(bool(r["article"]) for r in records) / count,
        "eval/format_rate": sum(r["format"]["format_reward"] for r in records) / count,
        "eval/assistant_turns": sum(r["assistant_turns"] for r in records) / count,
        "eval/response_tokens": sum(len(r["response_tokens"]) for r in records) / count,
        "eval/action_tokens": sum(sum(r["response_mask"]) for r in records) / count,
        "eval/tool_calls": calls / count,
        "eval/tool_error_rate": sum(r["tool_errors"] for r in records) / max(calls, 1),
        **{f"eval/finish_{reason}": finishes[reason] / count for reason in FINISH_REASONS},
    }


def generate(args):
    """每题贪心生成 1 条带工具调用的轨迹，最终 <answer> 就是待评分的报告。"""
    tasks = load_jsonl(args.bench_dir / "query.jsonl")
    if args.limit:
        tasks = tasks[:args.limit]
    tools = ResearchTools.from_env(BASE_DIR / ".env")
    rollout = RolloutConfig(group_size=1, temperature=0.0, seed=args.seed)
    config = {"questions": len(tasks), "base_model": args.base_model, "model_path": args.model_path,
              "batch_size": args.batch_size, "rollout": asdict(rollout), "search_count": tools.search_count}
    write_json(args.output_dir / "config.json", config)

    sampler = trio.ServiceClient().create_sampling_client(base_model=args.base_model, model_path=args.model_path)
    tokenizer = sampler.get_tokenizer()
    path = args.output_dir / "trajectories.jsonl"
    records = load_jsonl(path)
    done = {r["id"] for r in records}

    swanlab.init(project="tournament-grpo", name=args.run_name, config=config, mode=args.swanlab_mode)
    started = perf_counter()
    with path.open("a", encoding="utf-8") as file, \
            tqdm(total=len(tasks), initial=len(done), desc="Generate", unit="report") as progress:
        for offset in range(0, len(tasks), args.batch_size):
            batch = [t for t in tasks[offset:offset + args.batch_size] if str(t["id"]) not in done]
            if not batch:
                continue
            examples = [ResearchExample(str(t["id"]), t["prompt"], [], "deep_research_bench") for t in batch]
            trajectories = rollout_batch(sampler, tokenizer, tools, examples,
                                         replace(rollout, seed=args.seed + offset), lambda n: None)
            for task, trajectory in zip(batch, trajectories, strict=True):
                text = transcript_text(tokenizer, trajectory.prompt_tokens, trajectory.response_tokens)
                info = analyze_format(text)
                # 没有合法 <answer> 时 article 为空字符串，这道题仍然送去评分。
                record = {**asdict(trajectory), "id": str(task["id"]), "prompt": task["prompt"],
                          "language": task["language"], "article": info["rollout_answer"],
                          "response_text": text, "format": info}
                append_jsonl(file, record)
                records.append(record)
                progress.update(1)
            swanlab.log(generation_metrics(records), step=len(records))

    with (args.output_dir / "articles.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            append_jsonl(file, {key: record[key] for key in ("id", "prompt", "language", "article")})
    summary = {**generation_metrics(records), "eval/seconds_this_run": perf_counter() - started}
    write_json(args.output_dir / "generation-summary.json", summary)
    swanlab.log(summary, step=len(records) + 1)
    print(json.dumps(summary, indent=2))


def score(args):
    """每份报告和同题参考报告一起交给裁判，按作者的 RACE 规则算出相对分数。"""
    articles = load_jsonl(args.output_dir / "articles.jsonl")
    references = {r["prompt"]: r for r in load_jsonl(args.bench_dir / "reference.jsonl")}
    criteria = {r["prompt"]: r for r in load_jsonl(args.bench_dir / "criteria.jsonl")}
    judge = RaceJudge.from_env(BASE_DIR / ".env")
    score_dir = args.output_dir / "race" / path_name(judge.model)
    score_dir.mkdir(parents=True, exist_ok=True)
    config = {"judge_url": judge.api_url, "judge_model": judge.model, "judge_concurrency": args.judge_concurrency}
    write_json(score_dir / "score-config.json", config)

    path = score_dir / "race-results.jsonl"
    results = load_jsonl(path)
    done = {r["id"] for r in results}
    todo = [a for a in articles if a["id"] not in done]

    swanlab.init(project="tournament-grpo", name=f"{args.run_name}-{path_name(judge.model)}",
                 config=config, mode=args.swanlab_mode)
    with ThreadPoolExecutor(max_workers=args.judge_concurrency) as pool, \
            path.open("a", encoding="utf-8") as file, \
            tqdm(total=len(articles), initial=len(done), desc="RACE", unit="report") as progress:
        futures = [pool.submit(judge.score, a, references[a["prompt"]], criteria[a["prompt"]]) for a in todo]
        try:
            for future in as_completed(futures):
                result = future.result()
                append_jsonl(file, result)
                results.append(result)
                progress.update(1)
                swanlab.log({"race/completed": len(results)}, step=len(results))
        except Exception:
            # 某题重试 3 次仍失败：取消还在排队的请求，已写入的结果下次重跑时跳过。
            pool.shutdown(cancel_futures=True)
            raise

    summary = {f"race/{dim}": 100 * sum(r[dim] for r in results) / len(results)
               for dim in (*RACE_DIMS, "overall_score")}
    summary["race/questions"] = len(results)
    write_json(score_dir / "race-summary.json", summary)
    swanlab.log(summary, step=len(results) + 1)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "score"))
    parser.add_argument("--bench-dir", type=Path, default=BASE_DIR / "datasets/deep_research_bench")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs/drb",
                        help="输出根目录，自动追加 policy 模型名和权重标识")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--checkpoint", type=Path, help="训练保存的 checkpoint-step-N.json")
    parser.add_argument("--limit", type=int, default=0, help="只生成前 N 题；0 为全部 100 题")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--judge-concurrency", type=int, default=8, help="同时进行的 RACE 评分请求数")
    parser.add_argument("--run-name", help="SwanLab 运行名，默认包含 policy 模型、权重和阶段")
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    args.model_path = None
    if args.checkpoint:
        checkpoint = json.loads(args.checkpoint.read_text())
        args.base_model = checkpoint["config"]["base_model"]
        args.model_path = checkpoint["sampler_path"]
    weights = path_name(args.model_path) if args.model_path else "base"
    model = path_name(args.base_model)
    args.output_dir = args.output_dir / model / weights
    args.run_name = args.run_name or f"{model}-{weights}-{args.stage}"
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "generate":
        generate(args)
    else:
        score(args)
    swanlab.finish()
