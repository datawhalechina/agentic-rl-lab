"""PyTRIO 异步版 RGSD（Rubric-Guided Self-Distillation）。

骨架沿用 ``experiments/opsd/01-opsd-async.py``，区别在于：

- Teacher 的特权上下文是 rubric（论文附录 C.1 模板），而不是参考解答；
- 损失是 JSD_β（β=0.5）的 sampled-token 无偏估计，见 ``loss.py``；
- 开启 thinking 时，``</think>`` 及之前的 token 不进入损失（论文 B.1）。

每一步：

    1. 刷新 Student sampler，128 道题各采样 1 条回答（Student 只看问题）；
    2. 固定的 step-0 Teacher 看「问题 + rubric」，对同一条回答逐 token 打分；
    3. 由两边 logprob 算出逐 token advantage；
    4. 更新前检查（采样器与训练端一致、参数确实在变、截断率、数值）；
    5. 一次 forward_backward + 一次 optim_step。

只采样、不训练的检查（不创建 training run）：

    uv run python experiments/rgsd/train.py --domain medical --dry-run

readme 中 20 步实验的命令：

    uv run python experiments/rgsd/train.py \
        --domain medical --steps 20 --batch-size 128 --max-tokens 2048 \
        --learning-rate 4e-5 --save-every-steps 10 --max-concurrency 64 \
        --max-truncated-rate 0.4 --run-name rgsd-stageA-medical-lr4e-5-v3

续训时加上 ``--run-name <原 run 名> --resume-from <state_path> --start-step <已完成步数>``，
state_path 见 ``train-results/<run>/checkpoints.jsonl``。每步全部 rollout 会写入同目录的
``rollouts.jsonl``。
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm

from loss import sequence_advantages
from prompts import THINK_CLOSE, render_chat_prompt, student_message, teacher_message
from sampling import build_sampling_params, decode_completion, describe_answer


trio.configure(request_timeout_seconds=1800)

SCRIPT_DIR = Path(__file__).resolve().parent
LOSS_FN = "importance_sampling"
DRY_RUN_PROMPTS = 4

# JSD 估计量要求按 Student 自身分布无截断采样，所以采样参数固定，不开放成命令行参数。
TEMPERATURE, TOP_P, TOP_K = 1.0, 1.0, -1

# 论文用 HF Trainer 默认的 AdamW；grad clip 0.1 同论文。
ADAM = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.0, "grad_clip_norm": 0.1}


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 数据与模型。论文是全参微调，PyTRIO 提供的是 LoRA。
    parser.add_argument("--domain", choices=["medical", "science"], default="medical")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # 训练长度：--steps 优先；为 0 时按 --epochs 计算总步数。
    parser.add_argument("--steps", type=int, default=0, help="训练步数；0 表示按 --epochs 计算")
    parser.add_argument("--epochs", type=float, default=1.0, help="论文为 5")
    parser.add_argument("--batch-size", type=int, default=128, help="每步题数，每题 1 条 rollout（论文 128）")
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048, help="论文：instruct 2048 / thinking 4096")
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=False,
        help="Student 与 Teacher 是否开启 thinking；开启时思考部分不进入损失",
    )

    # 损失与优化器。
    parser.add_argument("--beta", type=float, default=0.5, help="JSD 混合系数，论文 0.5")
    parser.add_argument(
        "--learning-rate", type=float, default=4e-5,
        help="论文全参微调为 4.2e-6；LoRA 通常需要约 10 倍",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--warmup-steps", type=int, default=-1,
        help="显式指定 warmup 步数，覆盖 --warmup-ratio；冒烟测试用它对齐正式训练的学习率曲线",
    )

    # 更新前检查，任何一项不通过都会中止训练；设为 0 表示不检查。
    parser.add_argument(
        "--min-grad-norm", type=float, default=1e-3,
        help="grad_norm 低于该值就中止：此时每个参数的梯度远小于 Adam eps，等于没有更新",
    )
    parser.add_argument(
        "--max-logprob-gap", type=float, default=0.1,
        help="单条序列上采样器 logprob 与训练端 logprob 的平均绝对差阈值，超过就不参与本步更新",
    )
    parser.add_argument(
        "--max-gap-seq-rate", type=float, default=0.05,
        help="超过 --max-logprob-gap 的序列占比上限，超过就在更新前中止",
    )
    parser.add_argument(
        "--min-probe-delta", type=float, default=1e-4,
        help="固定探针序列的训练端 logprob 相对第一步的最小变化，低于它说明参数没在变",
    )
    parser.add_argument(
        "--max-truncated-rate", type=float, default=0.0,
        help="截断率超过该值时不做这一步更新，保存状态后以退出码 3 停止",
    )

    # 保存、续训与日志。
    parser.add_argument("--save-every-steps", type=int, default=50, help="0 表示只在最后保存")
    parser.add_argument("--resume-from", default=None, help="save_state 返回的 trio:// 路径")
    parser.add_argument("--start-step", type=int, default=0, help="续训时已完成的步数")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-rgsd")
    parser.add_argument("--swanlab-mode", choices=["online", "local", "offline", "disabled"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="只用 base model 跑几道题，不创建 training run")

    args = parser.parse_args()
    if not 0.0 < args.beta < 1.0:
        raise ValueError("--beta 必须在 (0, 1) 之间")
    if args.start_step > 0 and not args.resume_from:
        raise ValueError("--start-step 需要配合 --resume-from 使用")
    args.run_name = args.run_name or f"rgsd-{args.domain}-b{args.batch_size}-lr{args.learning_rate:g}"
    return args


# ---------------------------------------------------------------------------
# 数据与学习率
# ---------------------------------------------------------------------------


def load_rows(domain: str) -> list[dict[str, Any]]:
    path = SCRIPT_DIR / "datasets" / f"rubrichub_{domain}_train.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到训练数据：{path}\n请先运行：uv run python experiments/rgsd/prepare_data.py --domains {domain}"
        )
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def batch_indices(step: int, batch_size: int, num_rows: int, total_examples: int, seed: int) -> list[int]:
    """第 step 步用哪些题。每个 epoch 用 seed+epoch 重新打乱一次。

    结果只由 (seed, step) 决定，所以续训从 --start-step 开始就能接上原来的顺序。
    """
    orders: dict[int, list[int]] = {}
    indices = []
    for position in range(step * batch_size, min((step + 1) * batch_size, total_examples)):
        epoch, offset = divmod(position, num_rows)
        if epoch not in orders:
            orders[epoch] = list(range(num_rows))
            random.Random(seed + epoch).shuffle(orders[epoch])
        indices.append(orders[epoch][offset])
    return indices


def learning_rate_at(step: int, total_steps: int, args: argparse.Namespace) -> float:
    """线性 warmup 后保持常数（论文：constant + warmup，warmup 比例 0.1）。"""
    warmup = args.warmup_steps if args.warmup_steps >= 0 else math.ceil(args.warmup_ratio * total_steps)
    if warmup == 0:
        return args.learning_rate
    return args.learning_rate * min(1.0, (step + 1) / warmup)


# ---------------------------------------------------------------------------
# Rollout：Student 采样 → Teacher 打分 → 逐 token advantage
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    prompt_id: str
    prompt_ids: list[int]  # Student 的 prompt（只有问题）
    teacher_prompt_tokens: int
    completion_ids: list[int]
    student_logprobs: np.ndarray
    teacher_logprobs: np.ndarray
    train_mask: np.ndarray  # 1 表示该 token 进入损失
    advantages: np.ndarray  # 已乘 mask，并在序列内按 token 取平均
    raw_advantages: np.ndarray  # 未归一化的逐 token JSD advantage，只用于记录指标
    stop_reason: str
    answer: dict[str, Any]  # describe_answer 的结果：回答全文、长度、是否提到 rubric

    @property
    def trained(self) -> np.ndarray:
        return self.train_mask > 0


def build_train_mask(completion_ids: list[int], think_close_id: int | None) -> np.ndarray:
    """关闭 thinking 时全部 token 都训练；开启时只训练最后一个 ``</think>`` 之后的部分。

    开启 thinking 但没有闭合标签，说明还在思考阶段就被截断了，整条都不训练。
    """
    mask = np.ones(len(completion_ids))
    if think_close_id is None:
        return mask
    positions = [i for i, token in enumerate(completion_ids) if token == think_close_id]
    if not positions:
        return np.zeros_like(mask)
    mask[: positions[-1] + 1] = 0.0
    return mask


async def run_rollout(
    row: dict[str, Any],
    student_sampler: Any,
    teacher_client: Any,
    tokenizer: Any,
    params: trio.SamplingParams,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> Rollout | None:
    """单题：Student 只看问题采样一次 → Teacher 带 rubric 对同一条回答打分。"""
    prompt_ids = render_chat_prompt(tokenizer, student_message(row["question"]), args.enable_thinking)
    teacher_prompt_ids = render_chat_prompt(
        tokenizer, teacher_message(row["question"], row["rubrics"]), args.enable_thinking
    )

    async with semaphore:
        result = await student_sampler.sample_async(
            prompt=trio.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=params,
            return_text=False,
        )
    sequence = result.sequences[0]
    completion_ids = list(sequence.tokens)
    if not completion_ids:
        return None

    # Teacher 只对 Student 实际生成的回答打分，取 prompt 之后那一段的 logprob。
    async with semaphore:
        teacher_all = await teacher_client.compute_logprobs_async(
            trio.ModelInput.from_ints(teacher_prompt_ids + completion_ids)
        )
    teacher_lps = teacher_all[len(teacher_prompt_ids):]
    student_lps = sequence.logprobs
    for name, values in (("Student", student_lps), ("Teacher", teacher_lps)):
        if len(values) != len(completion_ids) or any(v is None for v in values):
            raise ValueError(f"{name} 的 logprob 与 completion token 无法一一对齐")

    think_close_id = tokenizer.convert_tokens_to_ids(THINK_CLOSE) if args.enable_thinking else None
    train_mask = build_train_mask(completion_ids, think_close_id)
    student_lps = np.asarray(student_lps, dtype=np.float64)
    teacher_lps = np.asarray(teacher_lps, dtype=np.float64)
    advantages, raw = sequence_advantages(student_lps, teacher_lps, train_mask, args.beta)
    text = decode_completion(tokenizer, completion_ids)
    return Rollout(
        prompt_id=row["id"],
        prompt_ids=prompt_ids,
        teacher_prompt_tokens=len(teacher_prompt_ids),
        completion_ids=completion_ids,
        student_logprobs=student_lps,
        teacher_logprobs=teacher_lps,
        train_mask=train_mask,
        advantages=advantages,
        raw_advantages=raw,
        stop_reason=sequence.stop_reason,
        answer=describe_answer(text, tokenizer, args.enable_thinking),
    )


async def collect_rollouts(rows, student_sampler, teacher_client, tokenizer, params, args, semaphore, desc):
    """并发完成一个 batch 的 rollout；同一 batch 全部来自同一个 Student checkpoint。"""
    with tqdm(total=len(rows), desc=desc, unit="prompt") as progress:

        async def one(row: dict[str, Any]) -> Rollout | None:
            rollout = await run_rollout(
                row, student_sampler, teacher_client, tokenizer, params, args, semaphore
            )
            progress.update(1)
            return rollout

        results = await asyncio.gather(*(one(row) for row in rows))
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# 构造 importance_sampling 的 Datum
# ---------------------------------------------------------------------------


def batch_scale(rollouts: list[Rollout]) -> float:
    """advantage 的缩放系数：batch 内平均每条序列的训练 token 数。

    PyTRIO 的 importance_sampling 对 token 求和、不做平均。advantage 已经在每条序列内
    按 token 取过平均，如果再按 batch 的总 token 数平均，batch 128、约 900 token 时梯度
    会缩小约 1e5 倍（实测 grad_norm 约 3e-7），比 Adam 的 eps 还小，等于学习率为 0。
    乘回平均长度后，梯度量级和仓库 OPSD 的逐 token 求和一致，长短回答的相对权重不变。
    """
    return float(np.mean([r.train_mask.sum() for r in rollouts]))


def build_datum(rollout: Rollout, scale: float) -> trio.Datum:
    """右移对齐：输入是 prompt + completion[:-1]，prompt 区间的 target / logprob / advantage 都填 0。

    scale=0 时 advantage 全为 0，这样的 Datum 只用来前向读取训练端 logprob。
    """
    pad = len(rollout.prompt_ids) - 1
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(rollout.prompt_ids + rollout.completion_ids[:-1]),
        loss_fn_inputs={
            "target_tokens": np.asarray([0] * pad + rollout.completion_ids, dtype=np.int64),
            "logprobs": np.concatenate([np.zeros(pad), rollout.student_logprobs]).astype(np.float32),
            "advantages": np.concatenate([np.zeros(pad), rollout.advantages * scale]).astype(np.float32),
        },
    )


# ---------------------------------------------------------------------------
# 更新前检查
# ---------------------------------------------------------------------------


class TruncationStop(Exception):
    """截断率超过 --max-truncated-rate：已保存状态，停下来交给人工判断。"""


async def trainer_logprobs(training_client: Any, rollouts: list[Rollout]) -> list[np.ndarray]:
    """只前向、不反向，取训练端在每条 completion 上的 logprob。"""
    datums = [build_datum(r, 0.0) for r in rollouts]
    result = await (await training_client.forward_async(datums, loss_fn=LOSS_FN))
    outputs = list(result.loss_fn_outputs)[: len(rollouts)]
    if len(outputs) != len(rollouts):
        raise RuntimeError(f"训练端只返回了 {len(outputs)} 条输出，少于提交的 {len(rollouts)} 条")
    values = []
    for rollout, output in zip(rollouts, outputs):
        if "logprobs" not in output:
            raise RuntimeError("前向输出里没有 logprobs，无法做更新前检查")
        logprobs = np.asarray(output["logprobs"].data, dtype=np.float64).reshape(-1)
        start = len(rollout.prompt_ids) - 1
        logprobs = logprobs[start : start + len(rollout.completion_ids)]
        if len(logprobs) != len(rollout.completion_ids):
            raise RuntimeError(f"[{rollout.prompt_id}] 训练端 logprob 长度与 completion 不一致")
        values.append(logprobs)
    return values


def is_inconsistent(gaps: np.ndarray, threshold: float) -> np.ndarray:
    """逐条判定采样器与训练端是否不一致；NaN / inf 一律算不一致，避免绕过检查。"""
    return ~np.isfinite(gaps) | (gaps > threshold)


def require_finite(name: str, value: Any) -> float:
    """关键指标缺失或不是有限值时直接中止。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"{name} 缺失或不是数值：{value!r}") from None
    if not math.isfinite(number):
        raise RuntimeError(f"{name} 不是有限值：{number}")
    return number


@dataclass
class Probe:
    """固定探针：取第一步的一条回答，之后每步都看它的训练端 logprob 有没有变。

    每次都单独前向：同一批里其他序列会影响数值，混在一起测会把批次差异误当成参数变化。
    """

    rollout: Rollout
    first: np.ndarray | None = None  # 第一步（更新前）的训练端 logprob
    noise: float = 0.0  # 同一权重前向两次的差，作为噪声底

    async def delta(self, training_client: Any) -> tuple[float, dict[str, float]]:
        """返回当前 logprob 相对第一步的平均绝对变化；第一次调用时顺便估计噪声。"""
        (now,) = await trainer_logprobs(training_client, [self.rollout])
        if not np.all(np.isfinite(now)):
            raise RuntimeError("探针 logprob 含非有限值")
        metrics: dict[str, float] = {}
        if self.first is None:
            (again,) = await trainer_logprobs(training_client, [self.rollout])
            self.first = now
            self.noise = float(np.abs(again - now).mean())
            metrics["probe/noise"] = self.noise
        delta = float(np.abs(now - self.first).mean())
        metrics["probe/delta_from_first_step"] = delta
        return delta, metrics


# ---------------------------------------------------------------------------
# 指标与记录
# ---------------------------------------------------------------------------


def divergence_stats(student: np.ndarray, teacher: np.ndarray, raw: np.ndarray, beta: float) -> dict[str, float]:
    """在被训练的 token 上统计散度。

    -A/(1-β) = log(π_S/m)，它在 π_S 下的均值就是 KL(π_S‖m) 的单样本估计。
    """
    return {
        "kl_student_mixture": float(np.mean(-raw / (1.0 - beta))),
        "reverse_kl_sampled": float(np.mean(student - teacher)),
        "teacher_prefers_rate": float(np.mean(teacher > student)),
    }


def rollout_metrics(rollouts: list[Rollout], beta: float) -> dict[str, float]:
    """一个 batch 的长度、截断、泄漏和散度指标；散度把所有被训练的 token 放在一起统计。"""
    metrics = {
        "data/rollouts": len(rollouts),
        "data/trained_rollouts": sum(bool(r.trained.any()) for r in rollouts),
        "data/completion_tokens_mean": float(np.mean([len(r.completion_ids) for r in rollouts])),
        "data/trained_tokens_total": float(sum(r.train_mask.sum() for r in rollouts)),
        "data/answer_tokens_mean": float(np.mean([r.answer["answer_tokens"] for r in rollouts])),
        "data/truncated_rate": float(np.mean([r.stop_reason == "length" for r in rollouts])),
        "data/teacher_prompt_tokens_mean": float(np.mean([r.teacher_prompt_tokens for r in rollouts])),
        # Student 从没见过 rubric，回答里出现 "criterion 3" 之类说明在模仿 Teacher 的措辞。
        "leak/answer_mentions_rubric_rate": float(np.mean([r.answer["answer_mentions_rubric"] for r in rollouts])),
    }
    trained = [r for r in rollouts if r.trained.any()]
    if trained:
        student = np.concatenate([r.student_logprobs[r.trained] for r in trained])
        teacher = np.concatenate([r.teacher_logprobs[r.trained] for r in trained])
        raw = np.concatenate([r.raw_advantages[r.trained] for r in trained])
        stats = divergence_stats(student, teacher, raw, beta)
        metrics.update({f"rgsd/{key}": value for key, value in stats.items()})
        metrics.update({
            "rgsd/advantage_raw_mean": float(raw.mean()),
            "rgsd/advantage_raw_std": float(raw.std()),
            "rgsd/advantage_raw_min": float(raw.min()),
            "rgsd/advantage_raw_max": float(raw.max()),
            "rgsd/student_logprob_mean": float(student.mean()),
            "rgsd/teacher_logprob_mean": float(teacher.mean()),
        })
    return metrics


def write_rollouts(path: Path, step: int, rollouts: list[Rollout], gaps: dict[str, float], beta: float) -> None:
    """把一步的全部 rollout 追加写入 JSONL，用于事后分析回答怎么变化。"""
    with path.open("a", encoding="utf-8") as file:
        for r in rollouts:
            record = {
                "step": step,
                "id": r.prompt_id,
                "stop_reason": r.stop_reason,
                "completion_tokens": len(r.completion_ids),
                "trained_tokens": int(r.trained.sum()),
                "logprob_gap": gaps.get(r.prompt_id),
                **r.answer,
            }
            if r.trained.any():
                m = r.trained
                record.update(divergence_stats(r.student_logprobs[m], r.teacher_logprobs[m], r.raw_advantages[m], beta))
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def numeric_metrics(prefix: str, values: dict[str, Any]) -> dict[str, float]:
    """只保留 PyTRIO 返回的数值指标，并加上命名空间前缀。"""
    result = {}
    for key, value in values.items():
        try:
            result[f"{prefix}/{key}"] = float(value)
        except (TypeError, ValueError):
            pass
    return result


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------


async def save_checkpoint(training_client: Any, run_name: str, step: int, manifest: Path, use_swanlab: bool) -> None:
    """保存可续训的 state 和可采样的 weights，路径追加写入 checkpoints.jsonl。"""
    name = f"{run_name}-step{step:06d}"
    state = await (await training_client.save_state_async(name=name))
    weights = await (await training_client.save_weights_for_sampler_async(name=name))
    record = {
        "step": step,
        "state_path": state.path,
        "sampler_path": weights.path,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with manifest.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    tqdm.write(f"Saved step {step}: {weights.path}")
    if use_swanlab:
        swanlab.log(
            {"checkpoint/state_path": swanlab.Text(state.path), "checkpoint/sampler_path": swanlab.Text(weights.path)},
            step=step,
        )


# ---------------------------------------------------------------------------
# 干跑：不创建 training run
# ---------------------------------------------------------------------------


async def dry_run(args: argparse.Namespace, rows: list[dict[str, Any]], service_client: Any) -> None:
    """用 base model 同时当 Student 和 Teacher 跑几道题，检查对齐、advantage 和 Datum。"""
    client = await service_client.create_sampling_client_async(base_model=args.base_model)
    tokenizer = client.get_tokenizer()
    params = build_sampling_params(tokenizer, args.max_tokens, TEMPERATURE, TOP_P, TOP_K, args.seed)
    batch = [rows[i] for i in batch_indices(0, DRY_RUN_PROMPTS, len(rows), len(rows), args.seed)]
    rollouts = await collect_rollouts(
        batch, client, client, tokenizer, params, args, asyncio.Semaphore(args.max_concurrency), "RGSD dry-run"
    )
    print(json.dumps(rollout_metrics(rollouts, args.beta), ensure_ascii=False, indent=2))

    trained = [r for r in rollouts if r.trained.any()]
    if not trained:
        print("没有可训练的回答（thinking 模式下可能全部在思考阶段被截断）")
        return
    scale = batch_scale(trained)
    for rollout in trained:
        datum = build_datum(rollout, scale)
        assert datum.model_input.length == len(rollout.prompt_ids) + len(rollout.completion_ids) - 1
    sample = trained[0]
    print(f"\n[{sample.prompt_id}] 训练 token {int(sample.train_mask.sum())}/{len(sample.completion_ids)}")
    print(f"Student 回答（前 600 字符）：\n{sample.answer['answer'][:600]}")
    print("\nDry run 完成：没有创建 training run。")


# ---------------------------------------------------------------------------
# 训练主循环
# ---------------------------------------------------------------------------


def prepare_result_dir(args: argparse.Namespace, total_steps: int) -> Path:
    """创建 train-results/<run>/ 并写入配置；全新训练时拒绝覆盖上一次的结果。"""
    result_dir = SCRIPT_DIR / "train-results" / args.run_name
    if not args.resume_from and any((result_dir / f).exists() for f in ("checkpoints.jsonl", "rollouts.jsonl")):
        raise FileExistsError(
            f"{result_dir} 里已有上一次训练的结果。全新训练请换 --run-name；续训请给出 --resume-from 和 --start-step"
        )
    result_dir.mkdir(parents=True, exist_ok=True)
    config_name = f"config-resume-step{args.start_step}.json" if args.resume_from else "config.json"
    config = {**vars(args), "total_steps": total_steps, "adam": ADAM,
              "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K}}
    (result_dir / config_name).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result_dir


async def train(args: argparse.Namespace) -> None:
    rows = load_rows(args.domain)
    service_client = trio.ServiceClient()
    if args.dry_run:
        await dry_run(args, rows, service_client)
        return

    total_examples = args.steps * args.batch_size if args.steps > 0 else math.ceil(args.epochs * len(rows))
    total_steps = math.ceil(total_examples / args.batch_size)
    if args.start_step >= total_steps:
        raise ValueError(f"--start-step {args.start_step} >= 总步数 {total_steps}")
    result_dir = prepare_result_dir(args, total_steps)
    manifest = result_dir / "checkpoints.jsonl"
    rollouts_path = result_dir / "rollouts.jsonl"
    print(f"Run: {args.run_name} | 训练题 {len(rows):,} | step {args.start_step}->{total_steps} | batch {args.batch_size}")

    if args.resume_from:
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(args.resume_from)
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=args.base_model, rank=args.lora_rank, seed=args.seed,
            train_attn=True, train_mlp=True, train_unembed=False,
        )
    # 不传 model_path：Teacher 永远是 step-0 的 base model，不随 Student 更新。
    teacher_client = await service_client.create_sampling_client_async(base_model=args.base_model)
    tokenizer = teacher_client.get_tokenizer()
    params = build_sampling_params(tokenizer, args.max_tokens, TEMPERATURE, TOP_P, TOP_K, args.seed)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    use_swanlab = args.swanlab
    if use_swanlab:
        swanlab.init(
            project=args.swanlab_project, name=args.run_name, mode=args.swanlab_mode,
            config={**vars(args), "total_steps": total_steps, "dataset_rows": len(rows)},
            tags=["PyTRIO", "RGSD", "RubricHub", args.domain],
            log_dir=str(SCRIPT_DIR / "swanlog"),
        )

    last_saved = args.start_step
    probe: Probe | None = None
    try:
        for step in range(args.start_step, total_steps):
            step_start = time.time()
            completed = step + 1

            # 1) 每步都把最新权重交给 sampler，保证 rollout 是严格 on-policy 的。
            student_sampler = await training_client.save_weights_and_get_sampling_client_async()
            batch = [rows[i] for i in batch_indices(step, args.batch_size, len(rows), total_examples, args.seed)]
            rollouts = await collect_rollouts(
                batch, student_sampler, teacher_client, tokenizer, params, args, semaphore,
                f"RGSD step {completed}/{total_steps}",
            )
            trained = [r for r in rollouts if r.trained.any()]
            if not trained:
                raise RuntimeError("本步没有任何可训练的回答")
            for r in trained:
                if not all(np.all(np.isfinite(x)) for x in (r.advantages, r.student_logprobs, r.teacher_logprobs)):
                    raise RuntimeError(f"[{r.prompt_id}] logprob 或 advantage 含非有限值，已在更新前中止")
            rollout_seconds = time.time() - step_start

            # 2) 更新前只做前向：比较采样器和训练端的 logprob，并读取固定探针。
            #    rollout 无论检查是否通过都会写入 rollouts.jsonl，方便事后排查。
            if probe is None:
                probe = Probe(trained[0])
            gaps = None
            try:
                if args.max_logprob_gap > 0:
                    trainer = await trainer_logprobs(training_client, trained)
                    gaps = np.array([np.abs(t - r.student_logprobs).mean() for t, r in zip(trainer, trained)])
                probe_delta, probe_metrics = (
                    await probe.delta(training_client) if args.min_probe_delta > 0 else (float("nan"), {})
                )
            finally:
                gap_by_id = {r.prompt_id: float(g) for r, g in zip(trained, gaps)} if gaps is not None else {}
                write_rollouts(rollouts_path, completed, rollouts, gap_by_id, args.beta)

            metrics: dict[str, Any] = {
                "train/step": completed,
                "time/rollout_seconds": rollout_seconds,
                **rollout_metrics(rollouts, args.beta),
                **probe_metrics,
            }
            if gaps is not None:
                metrics.update({
                    "consistency/seq_gap_mean": float(gaps.mean()),
                    "consistency/seq_gap_p50": float(np.percentile(gaps, 50)),
                    "consistency/seq_gap_p90": float(np.percentile(gaps, 90)),
                    "consistency/seq_gap_max": float(gaps.max()),
                    "consistency/seq_gap_over_threshold_rate": float(is_inconsistent(gaps, args.max_logprob_gap).mean()),
                })

            # 3) 探针：从第二步起，logprob 相对第一步的变化必须明显大于噪声。
            if args.min_probe_delta > 0 and step > args.start_step:
                need = max(args.min_probe_delta, 3.0 * probe.noise)
                if probe_delta < need:
                    raise RuntimeError(
                        f"step {completed}: 探针 [{probe.rollout.prompt_id}] 的 logprob 相对第一步只变化 "
                        f"{probe_delta:.2e}（需 > {need:.2e}）：参数没有实际更新，已中止"
                    )

            # 4) 截断率过高：不做这一步更新，保存已完成的状态后停止。
            truncated_rate = metrics["data/truncated_rate"]
            if args.max_truncated_rate > 0 and truncated_rate > args.max_truncated_rate:
                if use_swanlab:
                    swanlab.log(metrics, step=completed)
                if step > last_saved:
                    await save_checkpoint(training_client, args.run_name, step, manifest, use_swanlab)
                raise TruncationStop(
                    f"step {completed}: 截断率 {truncated_rate:.1%} > {args.max_truncated_rate:.0%}，"
                    f"未做该步更新；已完成 {step} 步的状态见 {manifest}"
                )

            # 5) 一致性：明显不一致的序列不参与更新；占比过高说明采样不是来自当前策略，直接中止。
            keep = trained
            if gaps is not None:
                bad = is_inconsistent(gaps, args.max_logprob_gap)
                if bad.mean() > args.max_gap_seq_rate:
                    raise RuntimeError(
                        f"step {completed}: {bad.mean():.1%} 的序列采样器与训练端 logprob 差超过 "
                        f"{args.max_logprob_gap}，已在更新前中止。逐条差值见 {rollouts_path}"
                    )
                keep = [r for r, b in zip(trained, bad) if not b]
            metrics["consistency/excluded_rollouts"] = len(trained) - len(keep)

            # 6) 更新：forward_backward 的 loss 必须是有限值，然后才做优化器步。
            lr = learning_rate_at(step, total_steps, args)
            scale = batch_scale(keep)
            fwd = await (await training_client.forward_backward_async(
                [build_datum(r, scale) for r in keep], loss_fn=LOSS_FN
            ))
            trainer_metrics = numeric_metrics("trainer", dict(fwd.metrics))
            for key in ("trainer/loss_sum", "trainer/loss_mean"):
                if key in trainer_metrics:
                    require_finite(key, trainer_metrics[key])
            optim = await (await training_client.optim_step_async(trio.AdamParams(learning_rate=lr, **ADAM)))

            metrics.update({
                "train/learning_rate": lr,
                "time/step_seconds": time.time() - step_start,
                **trainer_metrics,
                **numeric_metrics("optim", dict(optim.metrics)),
            })
            if use_swanlab:
                if args.save_every_steps and completed % args.save_every_steps == 0:
                    metrics["sample/student_answer"] = swanlab.Text(rollouts[0].answer["answer"])
                swanlab.log(metrics, step=completed)
            nan = float("nan")
            tqdm.write(
                f"step {completed:04d}/{total_steps} | lr {lr:.2e} | "
                f"kl(S||m) {metrics.get('rgsd/kl_student_mixture', nan):.4f} | "
                f"rev_kl {metrics.get('rgsd/reverse_kl_sampled', nan):.4f} | "
                f"gnorm {metrics.get('optim/grad_norm', nan):.2e} | "
                f"gap p50/max {metrics.get('consistency/seq_gap_p50', nan):.3f}/"
                f"{metrics.get('consistency/seq_gap_max', nan):.3f} | "
                f"excl {metrics['consistency/excluded_rollouts']} | probe Δ {probe_delta:.2e} | "
                f"tokens {metrics['data/completion_tokens_mean']:.0f} | trunc {truncated_rate:.2f} | "
                f"time {metrics['time/step_seconds']:.0f}s"
            )

            # 7) 更新后：散度和 grad_norm 必须是有限值；grad_norm 太小说明更新被 Adam eps 吞掉了。
            for key in ("rgsd/kl_student_mixture", "rgsd/reverse_kl_sampled"):
                require_finite(key, metrics.get(key))
            if args.min_grad_norm > 0:
                grad_norm = require_finite("optim/grad_norm", metrics.get("optim/grad_norm"))
                if grad_norm < args.min_grad_norm:
                    raise RuntimeError(
                        f"step {completed}: grad_norm={grad_norm:.3e} < {args.min_grad_norm:.1e}，"
                        "Adam 的更新会被 eps 主导，已中止以免浪费额度"
                    )

            if args.save_every_steps and completed % args.save_every_steps == 0:
                await save_checkpoint(training_client, args.run_name, completed, manifest, use_swanlab)
                last_saved = completed

        if last_saved != total_steps:
            await save_checkpoint(training_client, args.run_name, total_steps, manifest, use_swanlab)
        print(f"Completed {total_steps} RGSD steps. Checkpoints: {manifest}")
    except BaseException as error:
        if use_swanlab:
            if isinstance(error, (KeyboardInterrupt, TruncationStop)):
                swanlab.finish(state="aborted")
            else:
                swanlab.finish(state="crashed", error=str(error))
        raise
    if use_swanlab:
        swanlab.finish()


if __name__ == "__main__":
    started = time.time()
    try:
        asyncio.run(train(parse_args()))
    except TruncationStop as stop:
        print(f"Stopped: {stop}")
        raise SystemExit(3) from None
    print(f"All done in {time.time() - started:.2f}s")
