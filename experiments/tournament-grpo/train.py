"""用 PyTRIO 训练 Tournament-GRPO：淘汰赛奖励 + 格式奖励、组内标准化 advantage、作者的 PPO loss。

正式训练（仓库根目录）：
uv run python experiments/tournament-grpo/train.py \
    --max-steps 80 \
    --questions-per-batch 8 \
    --group-size 8 \
    --save-every 10 \
    --run-name tournament-grpo-qwen35-4b-80steps

单步冒烟测试（1 题 × 8 条候选，28 场比赛）：
uv run python experiments/tournament-grpo/train.py \
    --max-steps 1 \
    --questions-per-batch 1 \
    --run-name tournament-grpo-smoke

从 checkpoint 续训（--max-steps 是累计步数）：
uv run python experiments/tournament-grpo/train.py \
    --max-steps 80 \
    --resume experiments/tournament-grpo/outputs/<run-name>/checkpoint-step-40.json \
    --run-name <run-name>
"""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytrio as trio
import swanlab
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from data import load_examples
from judge import JudgeClient
from protocol import TOOL_BUDGET_NOTICE, transcript_text
from reward import score_group
from rollout import RolloutConfig, Trajectory, rollout_batch, trajectory_record
from search import ResearchTools


def assign_rewards(
    trajectories: list[Trajectory], tokenizer: Any, judge: JudgeClient,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """逐题跑淘汰赛，再按作者的组内标准化计算 advantage（样本标准差 ddof=1，eps=1e-6）。

    返回与 trajectories 对齐的奖励明细，以及每道题一条的 judge 日志。
    """
    groups = defaultdict(list)
    for index, trajectory in enumerate(trajectories):
        groups[trajectory.question_index].append(index)
    details, judge_logs = [{} for _ in trajectories], []
    for question_index, indices in groups.items():
        group = [trajectories[index] for index in indices]
        solutions = [transcript_text(tokenizer, t.prompt_tokens, t.response_tokens) for t in group]
        results, log = score_group(group[0].example, solutions, judge)
        rewards = torch.tensor([item["score"] for item in results])
        advantages = (rewards - rewards.mean()) / (rewards.std(correction=1) + 1e-6)
        for index, trajectory, result, advantage in zip(indices, group, results, advantages, strict=True):
            trajectory.reward, trajectory.advantage = result["score"], float(advantage)
            details[index] = result
        judge_logs.append({"question_index": question_index, **log, "advantages": advantages.tolist()})
    return details, judge_logs


def build_datum(trajectory: Trajectory) -> trio.Datum:
    """把 prompt + response 右移一位作为训练样本；mask 和 advantage 留在本地 loss 里使用。

    右移后第 i 个位置预测 tokens[i + 1]，所以 response 的 logprob 从 len(prompt) - 1 开始。
    """
    tokens = trajectory.prompt_tokens + trajectory.response_tokens
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": np.asarray(tokens[1:], dtype=np.int64)},
    )


def ppo_objective(
    current: torch.Tensor, old: torch.Tensor, advantages: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """作者的 vanilla PPO：token 平均、clip 0.2、dual clip 3，不加 KL 和熵项。"""
    log_ratio = (current - old).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    losses1 = -advantages * ratio
    losses2 = -advantages * ratio.clamp(0.8, 1.2)
    clipped = torch.maximum(losses1, losses2)
    dual_bound = -advantages * 3.0
    losses = torch.where(advantages < 0, torch.minimum(dual_bound, clipped), clipped)
    loss = losses.mean()
    return loss, {
        "ppo/loss": float(loss.detach()),
        "ppo/clipfrac": float((losses2 > losses1).float().mean()),
        "ppo/clipfrac_lower": float(((clipped > dual_bound) & (advantages < 0)).float().mean()),
        "ppo/kl": float(-log_ratio.detach().mean()),
    }


def make_ppo_loss_fn(trajectories: list[Trajectory]):
    """每批只更新一次：复用训练端前向的 detached logprob 作为 old，采样 logprob 只作诊断。

    若改成同批多次更新，必须在首次更新前固定 old，不能在每次回调里重新取值。
    """
    metas = [
        (len(t.prompt_tokens) - 1, t.response_mask, t.logprobs, t.advantage)
        for t in trajectories
    ]

    def loss_fn(data: list[trio.Datum], logprobs: list[torch.Tensor]):
        """logprobs 是训练端前向得到的逐 token logprob，顺序与 data 一致。"""
        current_values, sampling_values, advantage_values = [], [], []
        for values, (start, mask, sampling, advantage) in zip(logprobs, metas, strict=True):
            current = values.float().reshape(-1)
            action_mask = torch.tensor(mask, dtype=torch.bool, device=current.device)
            actions = current[start:][action_mask]
            current_values.append(actions)
            sampling_values.append(torch.tensor(sampling, device=current.device)[action_mask])
            advantage_values.append(torch.full_like(actions, advantage))
        current = torch.cat(current_values)
        loss, metrics = ppo_objective(current, current.detach(), torch.cat(advantage_values))
        metrics["ppo/sampling_logprob_absdiff"] = float((current.detach() - torch.cat(sampling_values)).abs().mean())
        metrics["ppo/action_tokens"] = float(current.numel())
        return loss, metrics

    return loss_fn


def update_policy(trainer: Any, trajectories: list[Trajectory], adam: trio.AdamParams) -> dict[str, float]:
    """每个 batch 只更新一次（1 个 epoch）；advantage 为 0 的组仍计入 token 平均。"""
    datums = [build_datum(t) for t in trajectories]
    loss_fn = make_ppo_loss_fn(trajectories)
    result = trainer.forward_backward_custom(datums, loss_fn).result()
    optimizer = trainer.optim_step(adam).result()
    return {**{f"trainer/{key}": value for key, value in result.metrics.items()},
            **{f"optimizer/{key}": value for key, value in optimizer.metrics.items()}}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def rollout_metrics(
    trajectories: list[Trajectory], details: list[dict[str, Any]], judge_logs: list[dict[str, Any]],
) -> dict[str, float]:
    """汇总奖励、judge、rollout 和工具指标，写入 train.jsonl 和 SwanLab。"""
    matches = [match for log in judge_logs for match in log["matches"]]
    usage = [match["output"]["usage"] for match in matches if isinstance(match["output"], dict)]
    calls = sum(len(t.tool_calls) for t in trajectories)
    metrics = {
        "reward/mean": mean([t.reward for t in trajectories]),
        "reward/tournament": mean([d["tournament_reward"] for d in details]),
        "reward/format": mean([d["format_reward"] for d in details]),
        "reward/has_answer": mean([float(bool(d["rollout_answer"])) for d in details]),
        "reward/degenerate_group_rate": mean([float(all(a == 0 for a in log["advantages"])) for log in judge_logs]),
        "judge/comparisons": float(len(matches)),
        # 只有 Jev 返回 usage；chat 后端的原始输出是字符串，这两项为 0。
        "judge/input_tokens": float(sum(u.get("input_tokens", 0) for u in usage)),
        "judge/output_tokens": float(sum(u.get("output_tokens", 0) for u in usage)),
        "rollout/assistant_turns": mean([t.assistant_turns for t in trajectories]),
        "rollout/response_tokens": mean([len(t.response_tokens) for t in trajectories]),
        "rollout/search_calls": mean([sum(c.name == "google_search" for c in t.tool_calls) for t in trajectories]),
        "rollout/browse_calls": mean([sum(c.name == "browse_webpage" for c in t.tool_calls) for t in trajectories]),
        "tools/error_rate": sum(t.tool_errors for t in trajectories) / max(calls, 1),
        "train/action_tokens": float(sum(sum(t.response_mask) for t in trajectories)),
    }
    finishes = Counter(t.finish_reason for t in trajectories)
    metrics.update({f"rollout/finish_{reason}": finishes[reason] / len(trajectories)
                    for reason in ("no_tool_call", "max_turns", "max_tool_calls", "max_response_tokens")})
    failures = Counter(reason for detail in details for reason in detail["failure_reasons"])
    metrics.update({f"format/{reason}": count / len(details) for reason, count in failures.items()})
    return metrics


def write_jsonl(path: Path, records: list[dict[str, Any]], mode: str = "a") -> None:
    with path.open(mode, encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(trainer: Any, output_dir: Path, step: int, config: dict[str, Any]) -> None:
    """保存可续训的 state 和可采样的权重，路径写进 checkpoint-step-N.json。"""
    state = trainer.save_state(name=f"step-{step}-state").result()
    weights = trainer.save_weights_for_sampler(name=f"step-{step}-weights").result()
    checkpoint = {"step": step, "state_path": state.path, "sampler_path": weights.path, "config": config}
    path = output_dir / f"checkpoint-step-{step}.json"
    path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tqdm.write(f"Saved checkpoint: {path} ({weights.path})")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-steps", type=int, required=True, help="累计训练步数")
    parser.add_argument("--data", type=Path, default=base / "datasets/train.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--questions-per-batch", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    # 作者全参训练用 1e-6；本仓库的 LoRA 章节统一用 4e-5。
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10, help="每 N 步保存一次；最后一步总会保存")
    parser.add_argument("--run-name", default="tournament-grpo-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--output-dir", type=Path, help="默认 experiments/tournament-grpo/outputs/<run-name>")
    parser.add_argument("--resume", type=Path, help="checkpoint-step-N.json：恢复权重、优化器和数据位置")
    parser.add_argument("--swanlab-project", default="tournament-grpo")
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline", "disabled"), default="online")
    args = parser.parse_args()
    args.output_dir = args.output_dir or base / "outputs" / args.run_name
    return args


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).resolve().parent
    # judge 和工具在创建训练 client 前读取配置，缺 Key 时立即报错。
    judge = JudgeClient.from_env(base / ".env")
    tools = ResearchTools.from_env(base / ".env")
    examples = load_examples(args.data)
    # 作者 shuffle=False、drop_last=True：按文件顺序取完整 batch，一个 epoch 后从头开始。
    batches_per_epoch = len(examples) // args.questions_per_batch
    checkpoint = json.loads(args.resume.read_text(encoding="utf-8")) if args.resume else None
    start_step = checkpoint["step"] if checkpoint else 0

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    service = trio.ServiceClient()
    if checkpoint:
        # state 同时包含 LoRA 权重和 AdamW 状态。
        trainer = service.create_training_client_from_state_with_optimizer(checkpoint["state_path"])
    else:
        trainer = service.create_lora_training_client(base_model=args.base_model, rank=args.lora_rank, seed=args.seed)
    adam = trio.AdamParams(
        learning_rate=args.learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, grad_clip_norm=1.0,
    )
    # 写入 SwanLab 和 checkpoint，记录与作者不同的后端和实现选择。
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "judge_backend": judge.backend,
        "judge_model": judge.model,
        "judge_long_context": judge.long_context,
        "optimizer": adam.model_dump(),
        "jina_reader_base_url": tools.jina_reader_base_url,
        "rollout": asdict(RolloutConfig(group_size=args.group_size, seed=args.seed)),
        "search_count": tools.search_count,
        "ppo_old_logprobs": "custom_forward_detached",
        "judge_failure_policy": "raise_after_3_attempts",
        "tool_budget_notice": TOOL_BUDGET_NOTICE,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    swanlab.init(project=args.swanlab_project, name=args.run_name, mode=args.swanlab_mode, config=config)
    try:
        with tqdm(total=args.max_steps, initial=start_step, desc="Training", unit="step") as progress:
            for step in range(start_step + 1, args.max_steps + 1):
                started = perf_counter()
                offset = ((step - 1) % batches_per_epoch) * args.questions_per_batch
                batch = examples[offset:offset + args.questions_per_batch]
                rollout_config = RolloutConfig(group_size=args.group_size, seed=args.seed + (step - 1) * args.questions_per_batch)

                # 1. 用当前权重采样：每道题 group_size 条多轮工具轨迹。
                progress.set_postfix(phase="rollout", refresh=True)
                sampler = trainer.save_weights_and_get_sampling_client()
                with tqdm(total=len(batch) * args.group_size, desc=f"Step {step} rollout", leave=False) as sampling:
                    trajectories = rollout_batch(sampler, tokenizer, tools, batch, rollout_config, sampling.update)
                sampled = perf_counter()

                # 2. 同题淘汰赛 + 格式奖励 → 组内 advantage。每场比赛的输入和 judge 输出都写进 judge.jsonl。
                progress.set_postfix(phase="judge", refresh=True)
                details, judge_logs = assign_rewards(trajectories, tokenizer, judge)
                judged = perf_counter()
                write_jsonl(args.output_dir / "judge.jsonl", [{"step": step, **log} for log in judge_logs])

                # 3. 一次 PPO 更新。
                progress.set_postfix(phase="update", refresh=True)
                metrics = {**rollout_metrics(trajectories, details, judge_logs), **update_policy(trainer, trajectories, adam)}
                updated = perf_counter()

                # 4. 保存 checkpoint；第一步和每个 checkpoint 保存完整轨迹，用于复盘模型行为的变化。
                save = step % args.save_every == 0 or step == args.max_steps
                if save:
                    progress.set_postfix(phase="checkpoint", refresh=True)
                    save_checkpoint(trainer, args.output_dir, step, config)
                if step == start_step + 1 or save:
                    records = [
                        trajectory_record(trajectory, tokenizer, step=step, reward_details=detail,
                                          base_model=args.base_model, config=asdict(rollout_config))
                        for trajectory, detail in zip(trajectories, details, strict=True)
                    ]
                    write_jsonl(args.output_dir / f"rollout-step-{step}.jsonl", records, mode="w")

                metrics.update({"time/rollout_seconds": sampled - started, "time/reward_seconds": judged - sampled,
                                "time/update_seconds": updated - judged, "time/step_seconds": perf_counter() - started})
                write_jsonl(args.output_dir / "train.jsonl", [{"step": step, "metrics": metrics}])
                swanlab.log(metrics, step=step)
                progress.update(1)
                progress.set_postfix(reward=f"{metrics['reward/mean']:.3f}",
                                     loss=f"{metrics['trainer/ppo/loss']:.4f}", refresh=True)
    except KeyboardInterrupt:
        swanlab.finish(state="aborted")
        raise
    except Exception as exc:
        swanlab.finish(state="crashed", error=str(exc))
        raise
    else:
        swanlab.finish()


if __name__ == "__main__":
    main(parse_args())
