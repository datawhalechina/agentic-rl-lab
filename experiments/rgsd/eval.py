"""RGSD 评测：生成与打分解耦。

``generate`` 只用 PyTRIO 采样，不需要任何 judge；``judge`` 读取生成结果，用指定的
judge 逐条判定 rubric（论文附录 C.2 的 prompt 与公式 1 的加权满足率）。分开的好处
是换 judge 时不必重新生成，不同 checkpoint 也能用同一个 judge 公平比较。

1) 生成。Base model：

    uv run python experiments/rgsd/eval.py generate --domain medical --tag base

   训练得到的 checkpoint（sampler_path 见 train-results/<run>/checkpoints.jsonl）：

    uv run python experiments/rgsd/eval.py generate --domain medical \
        --model-path 'trio://.../sampler_weights/...' --tag rgsd-step000050

   或一次生成某个 run 的全部 checkpoint：

    uv run python experiments/rgsd/eval.py generate --domain medical \
        --run-dir experiments/rgsd/train-results/<run>

2) 打分（judge 配置写在 experiments/rgsd/.env，见 .env.example）：

    uv run python experiments/rgsd/eval.py judge --judge-backend chat \
        experiments/rgsd/eval-results/generations-medical-eval-nothink-base.jsonl

提示词对照（readme 6.2 的 Base + thorough）：给 generate 加上
``--instruction "Provide a thorough response to the problem." --tag base-instr-thorough``，
指令会空一行追加在题目后面。

生成和打分都逐条追加落盘，中断后重跑同一命令会跳过已完成的条目。
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pytrio as trio
from tqdm import tqdm

from judge import build_judge, judge_response
from prompts import student_message
from sampling import build_sampling_params, generate_answer


trio.configure(sampling_timeout_seconds=3600)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "eval-results"


def parse_args() -> argparse.Namespace:
    """解析 generate / judge 两个子命令的参数。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="用 base model 或 checkpoint 生成回答")
    generate.add_argument("--domain", choices=["medical", "science"], default="medical")
    generate.add_argument("--split", choices=["train", "eval"], default="eval")
    generate.add_argument("--limit", type=int, default=0, help="只取前 N 题；0 表示全部")
    generate.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    source = generate.add_mutually_exclusive_group()
    source.add_argument("--model-path", default=None, help="单个 sampler weights 路径")
    source.add_argument(
        "--run-dir", type=Path, default=None,
        help="train.py 的结果目录；逐个生成 checkpoints.jsonl 里的全部 checkpoint",
    )
    generate.add_argument("--tag", default=None, help="输出文件名标签；默认按来源推断")
    generate.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=False
    )
    generate.add_argument(
        "--instruction",
        default=None,
        help="追加在问题后面的指令（空一行），用于对照实验，例如要求回答更全面；默认不加",
    )
    generate.add_argument("--max-tokens", type=int, default=2048)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--concurrency", type=int, default=32)

    judge = commands.add_parser("judge", help="用 judge 对生成结果逐条判定 rubric")
    judge.add_argument("generations", type=Path, nargs="+", help="generate 输出的 JSONL")
    judge.add_argument("--judge-backend", choices=["chat", "trio"], default="chat")
    judge.add_argument(
        "--judge-model", default="Qwen/Qwen3.8-27B", help="仅 trio 后端使用"
    )
    judge.add_argument("--judge-max-tokens", type=int, default=4096)
    judge.add_argument("--concurrency", type=int, default=16)
    judge.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "generate" and args.limit < 0:
        raise ValueError("--limit must be >= 0")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL；文件不存在时返回空列表。"""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_dataset_rows(domain: str, split: str, limit: int = 0) -> list[dict[str, Any]]:
    """读取 prepare_data.py 输出的 JSONL（已按固定 seed 打乱）。"""
    path = SCRIPT_DIR / "datasets" / f"rubrichub_{domain}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到数据：{path}\n请先运行：uv run python experiments/rgsd/"
            f"prepare_data.py --domains {domain}"
        )
    rows = read_jsonl(path)
    return rows[:limit] if limit else rows


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def generation_targets(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    """列出要生成的 (tag, model_path)；base model 的 model_path 为 None。"""
    if args.run_dir is not None:
        manifest = read_jsonl(args.run_dir / "checkpoints.jsonl")
        if not manifest:
            raise FileNotFoundError(f"{args.run_dir} 下没有 checkpoints.jsonl 记录")
        # 同一 step 被重复保存（例如续训）时只取最后一次。
        latest = {record["step"]: record["sampler_path"] for record in manifest}
        run = args.tag or args.run_dir.name
        return [(f"{run}-step{step:06d}", path) for step, path in sorted(latest.items())]
    if args.model_path:
        return [(args.tag or slugify(args.model_path.rsplit("/", 1)[-1]), args.model_path)]
    return [(args.tag or "base", None)]


async def generate_one_target(
    tag: str,
    model_path: str | None,
    rows: list[dict[str, Any]],
    service_client: Any,
    args: argparse.Namespace,
) -> None:
    """对一个模型来源生成全部回答，逐条追加写入 JSONL。"""
    thinking = "think" if args.enable_thinking else "nothink"
    output = RESULT_DIR / f"generations-{args.domain}-{args.split}-{thinking}-{tag}.jsonl"
    existing = read_jsonl(output)
    # 文件名不含生成参数；续写时参数必须与已有记录一致，否则两种设置的结果会混在同一个文件里。
    expected = {
        "model_path": model_path, "max_tokens": args.max_tokens, "temperature": args.temperature,
        "top_p": args.top_p, "enable_thinking": args.enable_thinking,
        "instruction": args.instruction,
    }
    for record in existing:
        mismatch = {k: (record.get(k), v) for k, v in expected.items() if record.get(k) != v}
        if mismatch:
            raise ValueError(
                f"{output.name} 里已有参数不同的记录 {record['id']}（已有值, 本次值）: {mismatch}；"
                "请换 --tag 写入新文件"
            )
    done = {record["id"] for record in existing}
    pending = [(index, row) for index, row in enumerate(rows) if row["id"] not in done]
    print(f"[{tag}] {len(rows) - len(pending)} done, {len(pending)} pending -> {output}")
    if not pending:
        return

    sampler = await service_client.create_sampling_client_async(
        base_model=args.base_model, model_path=model_path
    )
    tokenizer = sampler.get_tokenizer()
    semaphore = asyncio.Semaphore(args.concurrency)
    usage: Counter = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("a", encoding="utf-8") as file, tqdm(
        total=len(pending), desc=f"generate {tag}", unit="prompt"
    ) as progress:

        async def one(index: int, row: dict[str, Any]) -> None:
            # seed 只依赖题目下标，base 与各 checkpoint 在同一题上用同一个随机种子。
            params = build_sampling_params(
                tokenizer, args.max_tokens, args.temperature, args.top_p,
                seed=args.seed + index,
            )
            message = student_message(row["question"])
            if args.instruction:
                message = f"{message}\n\n{args.instruction.strip()}"
            generation = await generate_answer(
                sampler, tokenizer, message,
                enable_thinking=args.enable_thinking, params=params,
                semaphore=semaphore, usage=usage,
            )
            record = {
                "id": row["id"],
                "domain": args.domain,
                "split": args.split,
                "tag": tag,
                "base_model": args.base_model,
                "model_path": model_path,
                "enable_thinking": args.enable_thinking,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "instruction": args.instruction,
                **generation,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            progress.update(1)

        await asyncio.gather(*(one(index, row) for index, row in pending))
    print(f"[{tag}] usage: {dict(usage)}")


async def run_generate(args: argparse.Namespace) -> None:
    rows = load_dataset_rows(args.domain, args.split, args.limit)
    service_client = trio.ServiceClient()
    for tag, model_path in generation_targets(args):
        await generate_one_target(tag, model_path, rows, service_client, args)


def summarize_scores(records: list[dict[str, Any]], judge_name: str, seed: int) -> dict[str, Any]:
    """平均加权满足率（含 bootstrap 95% CI）、长度、截断与泄漏比例。"""
    scores = np.asarray([record["score"] for record in records])
    rng = np.random.default_rng(seed)
    resampled = rng.choice(scores, size=(2000, len(scores)), replace=True).mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    first = records[0]
    return {
        "tag": first["tag"],
        "domain": first["domain"],
        "split": first["split"],
        "model_path": first["model_path"],
        "judge": judge_name,
        "prompts": len(records),
        "score_pct": float(scores.mean() * 100),
        "score_pct_ci95": [float(low * 100), float(high * 100)],
        "answer_tokens_mean": float(np.mean([r["answer_tokens"] for r in records])),
        "completion_tokens_mean": float(np.mean([r["completion_tokens"] for r in records])),
        "truncated_rate": float(np.mean([r["stop_reason"] == "length" for r in records])),
        "empty_answer_rate": float(np.mean([not r["answer"] for r in records])),
        "answer_mentions_rubric_rate": float(
            np.mean([r["answer_mentions_rubric"] for r in records])
        ),
    }


async def judge_file(
    path: Path, judge: Any, args: argparse.Namespace, rubric_cache: dict[str, dict]
) -> None:
    """对一个生成文件逐条打分，写出 scores JSONL 与 summary JSON。"""
    generations = read_jsonl(path)
    if not generations:
        raise ValueError(f"生成结果为空：{path}")
    key = (generations[0]["domain"], generations[0]["split"])
    if key not in rubric_cache:
        rubric_cache[key] = {row["id"]: row for row in load_dataset_rows(*key)}
    rows = rubric_cache[key]

    judge_slug = slugify(judge.name)
    output = path.with_name(
        path.name.replace("generations-", "scores-", 1).removesuffix(".jsonl")
        + f"--{judge_slug}.jsonl"
    )
    done = {record["id"]: record for record in read_jsonl(output)}
    pending = [g for g in generations if g["id"] not in done]
    print(f"[{path.name}] {len(done)} scored, {len(pending)} pending -> {output.name}")

    semaphore = asyncio.Semaphore(args.concurrency)
    failed = 0
    usage: Counter = Counter()
    with output.open("a", encoding="utf-8") as file, tqdm(
        total=len(pending), desc=f"judge {generations[0]['tag']}", unit="answer"
    ) as progress:

        async def one(generation: dict[str, Any]) -> None:
            nonlocal failed
            row = rows[generation["id"]]
            result = await judge_response(
                judge, row["question"], generation["answer"], row["rubrics"], semaphore
            )
            if result is None:
                failed += 1
            else:
                usage["judge_input_tokens"] += result.input_tokens
                usage["judge_output_tokens"] += result.output_tokens
                record = {
                    **{k: v for k, v in generation.items() if k not in ("thinking",)},
                    "judge": judge.name,
                    "score": result.score,
                    "verdicts": result.verdicts,
                    "judge_attempts": result.attempts,
                }
                done[record["id"]] = record
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
            progress.update(1)

        await asyncio.gather(*(one(generation) for generation in pending))

    records = [done[g["id"]] for g in generations if g["id"] in done]
    summary = summarize_scores(records, judge.name, args.seed)
    summary["judge_failed_this_run"] = failed
    summary["usage_this_run"] = dict(usage)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{summary['tag']}] score {summary['score_pct']:.1f} "
        f"(95% CI {summary['score_pct_ci95'][0]:.1f}~{summary['score_pct_ci95'][1]:.1f}) | "
        f"answer tokens {summary['answer_tokens_mean']:.0f} | judged {len(records)} | "
        f"failed {failed} -> {summary_path.name}"
    )


async def run_judge(args: argparse.Namespace) -> None:
    service_client = trio.ServiceClient() if args.judge_backend == "trio" else None
    judge = await build_judge(
        args.judge_backend, service_client, args.judge_model, args.judge_max_tokens
    )
    rubric_cache: dict[tuple[str, str], dict] = {}
    for path in args.generations:
        await judge_file(path, judge, args, rubric_cache)


if __name__ == "__main__":
    parsed = parse_args()
    asyncio.run(run_generate(parsed) if parsed.command == "generate" else run_judge(parsed))
