"""RGSD 训练前诊断：rubric 条件增益（论文 Table 1）。

同一个基础模型在两种条件下回答同一批问题：

- ``base``：只看问题（就是 RGSD 的 Student 输入）；
- ``rubric``：问题 + rubric + transition prompt（就是 RGSD 的 Teacher 输入）。

再用同一个 judge 对两份回答逐条判定 rubric，比较加权满足率。两者之差就是 RGSD
可以蒸馏进 Student 的信号上限；差值很小时不值得继续训练。全程只做前向，不训练。

先准备数据：

    uv run python experiments/rgsd/prepare_data.py --domains medical

冒烟测试（4 题）：

    uv run python experiments/rgsd/conditioning_gap.py --domain medical --limit 4

完整诊断（固定 300 题评测集）：

    uv run python experiments/rgsd/conditioning_gap.py --domain medical

逐题结果追加写入 JSONL，中断后重跑同一条命令会跳过已完成的题目；汇总写入同名的
``.summary.json``。
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pytrio as trio
from tqdm import tqdm

from judge import build_judge, judge_response
from prompts import student_message, teacher_message
from sampling import build_sampling_params, generate_answer


trio.configure(sampling_timeout_seconds=3600)

SCRIPT_DIR = Path(__file__).resolve().parent
CONDITIONS = ("base", "rubric")


def parse_args() -> argparse.Namespace:
    """解析并校验诊断的数据、生成、judge 和输出参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["medical", "science"], default="medical")
    parser.add_argument(
        "--split",
        choices=["train", "eval"],
        default="eval",
        help="使用 prepare_data.py 切出的哪份数据；默认固定 300 题评测集",
    )
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 题；0 表示全部")
    parser.add_argument("--seed", type=int, default=42)

    # 被诊断的基础模型与生成参数；两种条件使用完全相同的采样设置。
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="生成时是否开启 thinking；judge 只看 </think> 之后的最终回答",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gen-concurrency", type=int, default=16)

    # judge：默认用 PyTRIO 上更大的模型，避免额外的 API key。
    parser.add_argument("--judge-backend", choices=["trio", "chat"], default="trio")
    parser.add_argument("--judge-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-concurrency", type=int, default=8)

    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    for name in ("max_tokens", "gen_concurrency", "judge_max_tokens", "judge_concurrency"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.output is None:
        thinking = "think" if args.enable_thinking else "nothink"
        args.output = (
            SCRIPT_DIR / "eval-results"
            / f"conditioning-gap-{args.domain}-{args.split}-{thinking}.jsonl"
        )
    return args


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """读取 prepare_data.py 生成的 JSONL。文件已按固定 seed 打乱，直接取前 N 条。"""
    path = SCRIPT_DIR / "datasets" / f"rubrichub_{args.domain}_{args.split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到数据：{path}\n请先运行：uv run python experiments/rgsd/"
            f"prepare_data.py --domains {args.domain}"
        )
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    if not rows:
        raise ValueError(f"数据为空：{path}")
    return rows


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    """读取已经完成的逐题结果，用于断点续跑。"""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return {record["id"]: record for record in map(json.loads, file)}


async def run_item(
    index: int,
    row: dict[str, Any],
    sampler: Any,
    tokenizer: Any,
    judge: Any,
    args: argparse.Namespace,
    gen_semaphore: asyncio.Semaphore,
    judge_semaphore: asyncio.Semaphore,
    usage: Counter,
) -> dict[str, Any] | None:
    """完成单题的两次生成和两次判定；judge 连续失败时返回 None。"""
    messages = {
        "base": student_message(row["question"]),
        "rubric": teacher_message(row["question"], row["rubrics"]),
    }
    # 两种条件共用同一个 seed，差异只来自上下文里有没有 rubric。
    params = build_sampling_params(
        tokenizer, args.max_tokens, args.temperature, args.top_p, seed=args.seed + index
    )
    generations = await asyncio.gather(
        *(
            generate_answer(
                sampler, tokenizer, messages[name],
                enable_thinking=args.enable_thinking, params=params,
                semaphore=gen_semaphore, usage=usage,
            )
            for name in CONDITIONS
        )
    )
    verdicts = await asyncio.gather(
        *(
            judge_response(
                judge, row["question"], generation["answer"], row["rubrics"],
                judge_semaphore,
            )
            for generation in generations
        )
    )
    # 两个条件必须成对有效，否则这道题不进入比较。
    if any(result is None for result in verdicts):
        return None

    record: dict[str, Any] = {
        "id": row["id"],
        "question": row["question"],
        "num_criteria": len(row["rubrics"]),
    }
    for name, generation, result in zip(CONDITIONS, generations, verdicts, strict=True):
        usage["judge_input_tokens"] += result.input_tokens
        usage["judge_output_tokens"] += result.output_tokens
        record[name] = {
            **generation,
            "score": result.score,
            "verdicts": result.verdicts,
            "judge_attempts": result.attempts,
        }
    return record


def summarize(
    records: list[dict[str, Any]], args: argparse.Namespace, judge_name: str
) -> dict[str, Any]:
    """聚合成对结果：两种条件的均分、增益及其 bootstrap 置信区间。"""
    scores = {
        name: np.asarray([record[name]["score"] for record in records])
        for name in CONDITIONS
    }
    lifts = scores["rubric"] - scores["base"]
    # 成对 bootstrap：对题目重采样，保留同一题两种条件之间的相关性。
    rng = np.random.default_rng(args.seed)
    resampled = rng.choice(lifts, size=(2000, len(lifts)), replace=True).mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])

    summary: dict[str, Any] = {
        "domain": args.domain,
        "split": args.split,
        "base_model": args.base_model,
        "enable_thinking": args.enable_thinking,
        "judge": judge_name,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pairs": len(records),
        "lift_pp": float(lifts.mean() * 100),
        "lift_pp_ci95": [float(low * 100), float(high * 100)],
        "rubric_better_rate": float((lifts > 0).mean()),
        "rubric_worse_rate": float((lifts < 0).mean()),
    }
    for name in CONDITIONS:
        items = [record[name] for record in records]
        summary[name] = {
            "score_pct": float(scores[name].mean() * 100),
            "answer_tokens_mean": float(np.mean([item["answer_tokens"] for item in items])),
            "completion_tokens_mean": float(
                np.mean([item["completion_tokens"] for item in items])
            ),
            "stop_reasons": dict(Counter(item["stop_reason"] for item in items)),
            "empty_answer_rate": float(np.mean([not item["answer"] for item in items])),
            "answer_mentions_rubric_rate": float(
                np.mean([item["answer_mentions_rubric"] for item in items])
            ),
            "thinking_mentions_rubric_rate": float(
                np.mean([item["thinking_mentions_rubric"] for item in items])
            ),
        }
    return summary


async def main(args: argparse.Namespace) -> None:
    """创建采样与 judge 客户端，并发完成诊断并落盘。"""
    rows = load_rows(args)
    done = load_done(args.output)
    pending = [(index, row) for index, row in enumerate(rows) if row["id"] not in done]
    print(f"Loaded {len(rows)} prompts ({args.domain}/{args.split}); pending {len(pending)}")

    usage: Counter = Counter()
    judge_name = f"{args.judge_backend}:{args.judge_model}"
    failed = 0
    if pending:
        service_client = trio.ServiceClient()
        sampler = await service_client.create_sampling_client_async(
            base_model=args.base_model
        )
        tokenizer = sampler.get_tokenizer()
        judge = await build_judge(
            args.judge_backend, service_client, args.judge_model, args.judge_max_tokens
        )
        judge_name = judge.name
        gen_semaphore = asyncio.Semaphore(args.gen_concurrency)
        judge_semaphore = asyncio.Semaphore(args.judge_concurrency)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        # 每完成一题立即追加落盘：采样和 judge 都是付费请求，中断后不应重跑已完成的题目。
        with (
            args.output.open("a", encoding="utf-8") as file,
            tqdm(total=len(pending), desc="conditioning gap", unit="prompt") as progress,
        ):

            async def run_and_save(index: int, row: dict[str, Any]) -> None:
                nonlocal failed
                record = await run_item(
                    index, row, sampler, tokenizer, judge, args,
                    gen_semaphore, judge_semaphore, usage,
                )
                if record is None:
                    failed += 1
                else:
                    done[record["id"]] = record
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    file.flush()
                progress.update(1)

            started = time.time()
            await asyncio.gather(*(run_and_save(index, row) for index, row in pending))
            print(f"Finished {len(pending)} prompts in {time.time() - started:.0f}s")

    records = [done[row["id"]] for row in rows if row["id"] in done]
    if not records:
        raise RuntimeError("没有任何成对有效的结果，无法汇总")
    summary = summarize(records, args, judge_name)
    summary["judge_failed_prompts"] = failed
    summary["usage_this_run"] = dict(usage)

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"base {summary['base']['score_pct']:.1f} | "
        f"+rubric {summary['rubric']['score_pct']:.1f} | "
        f"lift {summary['lift_pp']:+.1f}pp "
        f"(95% CI {summary['lift_pp_ci95'][0]:+.1f} ~ {summary['lift_pp_ci95'][1]:+.1f}) | "
        f"pairs {summary['pairs']} | judge failed {failed}"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {args.output}\nSummary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
