"""使用 PyTRIO 和可切换搜索后端训练 Qwen3.5-4B Search-R1。

在 experiments/search-r1 目录下使用 DeepSeek Search 进行正式训练：
uv run python train.py \
    --max-steps 100 \
    --questions-per-batch 8 \
    --group-size 8 \
    --save-every 50 \
    --base-model Qwen/Qwen3.5-4B \
    --search-backend deepseek \
    --run-name search-r1-qwen35-4b-deepseek-100steps \
    --swanlab-mode online

在 experiments/search-r1 目录下使用 DeepSeek Search 进行小规模训练：
uv run python train.py \
    --max-steps 30 \
    --questions-per-batch 8 \
    --group-size 8 \
    --save-every 10 \
    --base-model Qwen/Qwen3.5-4B \
    --search-backend deepseek \
    --run-name search-r1-qwen35-4b-deepseek \
    --swanlab-mode online

使用 Wikipedia 进行无需搜索 API Key 的小规模训练：
uv run python train.py \
    --max-steps 20 \
    --questions-per-batch 8 \
    --group-size 8 \
    --save-every 5 \
    --base-model Qwen/Qwen3.5-4B \
    --search-backend wikipedia \
    --run-name search-r1-qwen35-4b-wikipedia \
    --swanlab-mode online

"""

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm

from data import shuffled_examples, take_batch
from rollout import RolloutConfig, Trajectory, rollout_batch
from search import (
    SEARCH_BACKENDS,
    create_search_client,
    resolve_search_concurrency,
    resolve_search_timeout,
)


MAX_TRAIN_CONTEXT_TOKENS = 8192  # 单个训练 Datum 允许的最大 token 数。


def build_datum(trajectory: Trajectory) -> trio.Datum:
    """把一条前缀扩展的多轮轨迹合成一个右移对齐的训练 Datum。"""
    if not trajectory.turns:
        raise ValueError("不能用没有 assistant turn 的轨迹构造训练 Datum")

    full_tokens: list[int] = []
    old_logprobs_by_token: list[float] = []
    advantages_by_token: list[float] = []
    assistant_token_count = 0

    for turn_index, turn in enumerate(trajectory.turns):
        if len(turn.completion_tokens) != len(turn.logprobs):
            raise ValueError(
                f"第 {turn_index + 1} 个 assistant turn 的 token 与 logprob 长度不一致"
            )

        # 下一轮 prompt 应由已有轨迹加上新的 tool observation 和 assistant 前缀组成。
        if turn_index == 0:
            delta_observation = turn.prompt_tokens
        elif turn.prompt_tokens[: len(full_tokens)] == full_tokens:
            delta_observation = turn.prompt_tokens[len(full_tokens) :]
        else:
            raise ValueError(
                f"第 {turn_index + 1} 个 assistant turn 的 prompt "
                "不是已有轨迹的前缀扩展，无法安全对齐采样 logprob"
            )

        full_tokens.extend(delta_observation)
        full_tokens.extend(turn.completion_tokens)
        old_logprobs_by_token.extend([0.0] * len(delta_observation))
        old_logprobs_by_token.extend(turn.logprobs)
        advantages_by_token.extend([0.0] * len(delta_observation))
        advantages_by_token.extend(
            [trajectory.advantage] * len(turn.completion_tokens)
        )
        assistant_token_count += len(turn.completion_tokens)

    if assistant_token_count == 0:
        raise ValueError("不能用没有 assistant token 的轨迹构造训练 Datum")
    if not (
        len(full_tokens)
        == len(old_logprobs_by_token)
        == len(advantages_by_token)
    ):
        raise ValueError("完整轨迹的 token、logprob 和 advantage 长度不一致")

    # 对完整序列统一右移；observation 保留真实 target token，但训练信号为零。
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    old_logprobs = old_logprobs_by_token[1:]
    advantages = advantages_by_token[1:]
    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(old_logprobs)
        == len(advantages)
    ):
        raise ValueError("Datum 的 input、target、logprobs 和 advantages 长度不一致")
    if len(input_tokens) > MAX_TRAIN_CONTEXT_TOKENS:
        raise ValueError(f"Datum 超过 {MAX_TRAIN_CONTEXT_TOKENS} token")
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(old_logprobs, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
        },
    )


def mean(values: list[float]) -> float:
    """计算列表均值，空列表返回零。"""
    return sum(values) / len(values) if values else 0.0


def source_reward(trajectories: list[Trajectory], source_name: str) -> float:
    """计算指定训练来源的平均 reward。"""
    rewards = [
        trajectory.reward
        for trajectory in trajectories
        if source_name in trajectory.example.data_source.lower()
    ]
    return mean(rewards)


def degenerate_group_count(trajectories: list[Trajectory]) -> int:
    """统计整组 advantage 都为零的问题数。"""
    groups: dict[int, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory.question_index].append(trajectory.advantage)
    return sum(all(advantage == 0.0 for advantage in values) for values in groups.values())


def rollout_metrics(
    trajectories: list[Trajectory],
    datums: list[trio.Datum],
    question_count: int,
) -> dict[str, float]:
    """汇总 reward、轨迹和完整训练 batch 的 token 指标。"""
    tool_attempts = sum(
        "<tool_call>" in turn.text
        for trajectory in trajectories
        for turn in trajectory.turns
    )
    valid_tool_calls = sum(trajectory.search_calls for trajectory in trajectories)
    trajectory_lengths = [
        len(trajectory.turns[-1].prompt_tokens)
        + len(trajectory.turns[-1].completion_tokens)
        for trajectory in trajectories
        if trajectory.turns
    ]
    input_tokens = sum(len(datum.model_input) for datum in datums)
    loss_tokens = sum(
        np.count_nonzero(datum.loss_fn_inputs["advantages"].to_numpy())
        for datum in datums
    )
    return {
        "reward/mean": mean([trajectory.reward for trajectory in trajectories]),
        "reward/correct": mean([float(trajectory.exact_match) for trajectory in trajectories]),
        "reward/format": mean([float(trajectory.valid_format) for trajectory in trajectories]),
        "reward/nq": source_reward(trajectories, "nq"),
        "reward/hotpotqa": source_reward(trajectories, "hotpotqa"),
        "rollout/turns": mean([float(len(trajectory.turns)) for trajectory in trajectories]),
        "rollout/search_calls": mean(
            [float(trajectory.search_calls) for trajectory in trajectories]
        ),
        "rollout/trajectory_tokens": mean([float(value) for value in trajectory_lengths]),
        "rollout/valid_tool_call_rate": valid_tool_calls / max(tool_attempts, 1),
        "rollout/degenerate_group_rate": degenerate_group_count(trajectories)
        / max(question_count, 1),
        "train/datums_per_rollout_batch": float(len(datums)),
        "train/tokens_per_rollout_batch": float(input_tokens),
        "train/loss_tokens_per_rollout_batch": float(loss_tokens),
    }


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    """把 Path 参数转换成 SwanLab 可记录的字符串。"""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(training_client: Any, name: str) -> None:
    """同时保存断点续训 state 和推理 sampler weights。"""
    state = training_client.save_state(name=f"{name}-state").result()
    weights = training_client.save_weights_for_sampler(name=f"{name}-weights").result()
    print(f"Saved state: {state.path}")
    print(f"Saved sampler weights: {weights.path}")


def parse_args() -> argparse.Namespace:
    """解析训练超参数。"""
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
        help="最多执行多少个 GRPO 训练 step",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=base_dir / "datasets" / "train.jsonl",
        help="训练数据 JSONL 文件路径",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="最多使用多少条训练问题；0 表示使用全部数据",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.5-4B",
        help="创建 LoRA 训练客户端使用的基础模型",
    )
    parser.add_argument(
        "--resume-state",
        help="从 save_state() 保存的训练状态恢复；设置后不再新建 LoRA 客户端",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="新建 LoRA 训练客户端时使用的 rank",
    )
    parser.add_argument(
        "--questions-per-batch",
        type=int,
        default=8,
        help="每个训练 step 选取的问题数量",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="同一道问题采样的轨迹数量，用于计算组内相对 advantage",
    )
    parser.add_argument(
        "--max-search-calls",
        type=int,
        default=4,
        help="每条轨迹最多调用搜索工具的次数",
    )
    parser.add_argument(
        "--search-backend",
        choices=SEARCH_BACKENDS,
        default="deepseek",
        help="搜索后端：deepseek 官方 Search API，wikipedia 免费免密钥",
    )
    parser.add_argument(
        "--search-concurrency",
        type=int,
        default=None,
        help="不同轨迹之间的搜索并发；默认 deepseek=16、wikipedia=3",
    )
    parser.add_argument(
        "--search-count",
        type=int,
        choices=range(1, 51),
        default=3,
        metavar="1-50",
        help="DeepSeek Search 返回的网页条数；wikipedia 后端固定返回 Top 3",
    )
    parser.add_argument(
        "--search-timeout",
        type=float,
        default=None,
        help="单次搜索的超时秒数；默认 deepseek=60、wikipedia=15",
    )
    parser.add_argument(
        "--max-assistant-turns",
        type=int,
        default=6,
        help="每条轨迹最多生成的 assistant 回合数",
    )
    parser.add_argument(
        "--max-trajectory-tokens",
        type=int,
        default=8192,
        help="整条轨迹允许使用的最大 token 数",
    )
    parser.add_argument(
        "--max-assistant-tokens",
        type=int,
        default=1024,
        help="单个 assistant 回合最多生成的 token 数",
    )
    parser.add_argument(
        "--max-tool-response-tokens",
        type=int,
        default=1024,
        help="单次搜索结果最多保留的 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="rollout 采样温度；越高随机性越强",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="rollout 核采样的累积概率阈值",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=4e-5,
        help="Adam 优化器学习率",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Adam 优化器的一阶动量系数",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="Adam 优化器的二阶动量系数",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help=(
            "每隔多少个 step 同时保存断点续训 state 和推理 sampler weights；"
            "0 表示只在训练结束时保存"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="数据打乱、LoRA 初始化和 rollout 采样使用的随机种子",
    )
    parser.add_argument(
        "--run-name",
        default="search-r1-qwen35-4b",
        help="SwanLab 实验名称，同时作为 checkpoint 名称前缀",
    )
    parser.add_argument(
        "--swanlab-project",
        default="agentic-rl-lab-search-r1",
        help="SwanLab 项目名称",
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="online",
        help="SwanLab 日志模式",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """运行同步训练循环，并仅在 rollout 采样时使用 async。"""
    args.search_concurrency = resolve_search_concurrency(
        args.search_backend, args.search_concurrency
    )
    args.search_timeout = resolve_search_timeout(
        args.search_backend, args.search_timeout
    )
    search_client = create_search_client(
        args.search_backend,
        Path(__file__).resolve().parent / ".env",
        count=args.search_count,
        timeout=args.search_timeout,
    )

    # 先固定打乱训练问题；max_train_samples 只用于限制本次实际参与训练的数据量。
    examples = shuffled_examples(args.data, args.seed)
    if args.max_train_samples > 0:
        examples = examples[: args.max_train_samples]
    if not examples:
        raise ValueError("训练数据为空，请先运行 prepare_data.py")

    # 有 state 就恢复完整训练状态，否则从基础模型新建一个 LoRA 训练客户端。
    service_client = trio.ServiceClient()
    if args.resume_state:
        training_client = service_client.create_training_client_from_state(
            args.resume_state
        )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=args.base_model,
            rank=args.lora_rank,
            seed=args.seed,
        )

    # tokenizer 来自训练客户端，保证 rollout、训练 Datum 使用完全相同的分词方式。
    tokenizer = training_client.get_tokenizer()

    # 将命令行中的采样、搜索次数和轨迹长度限制集中成 rollout 配置。
    rollout_config = RolloutConfig(
        group_size=args.group_size,
        max_search_calls=args.max_search_calls,
        max_assistant_turns=args.max_assistant_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        max_tool_response_tokens=args.max_tool_response_tokens,
        search_concurrency=args.search_concurrency,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    # 每个训练 step 处理完整 rollout batch，并只 optim_step 一次。
    adam_params = trio.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    # 记录完整命令行配置，方便后续从 SwanLab 对照和复现实验。
    run = swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        mode=args.swanlab_mode,
        config=serializable_config(args),
    )

    try:
        # 外层进度条统计已完成的训练 step，并自动显示总耗时和预计剩余时间。
        with tqdm(
            total=args.max_steps,
            desc="Training",
            unit="step",
            position=0,
        ) as training_progress:
            for step in range(args.max_steps):
                step_started = perf_counter()

                # 按 questions_per_batch 循环取题；超过数据末尾时 take_batch 会回绕。
                batch = take_batch(
                    examples,
                    step * args.questions_per_batch,
                    args.questions_per_batch,
                )

                # 导出当前 LoRA 权重创建 sampler，确保本 step 的 rollout 来自当前策略。
                training_progress.set_postfix(phase="prepare sampler", refresh=True)
                sampling_client = training_client.save_weights_and_get_sampling_client()

                # 内层进度条显示当前 step 已完成多少条轨迹。
                training_progress.set_postfix(phase="rollout", refresh=True)
                with tqdm(
                    total=len(batch) * args.group_size,
                    desc=f"Step {step + 1}/{args.max_steps} rollout",
                    unit="trajectory",
                    position=1,
                    leave=False,
                ) as rollout_progress:
                    # 为每道题采样 group_size 条多轮搜索轨迹，并计算 reward 和组内 advantage。
                    trajectories = rollout_batch(
                        sampling_client=sampling_client,
                        tokenizer=tokenizer,
                        search_client=search_client,
                        examples=batch,
                        config=rollout_config,
                        progress_callback=rollout_progress.update,
                    )

                # 完整 rollout batch 一次提交，SDK 自动拆批、归一化并累积梯度。
                training_progress.set_postfix(phase="build datums", refresh=True)
                datums = [build_datum(trajectory) for trajectory in trajectories]
                metrics = rollout_metrics(trajectories, datums, len(batch))
                update_skipped = not any(
                    trajectory.advantage != 0.0 for trajectory in trajectories
                )
                if not update_skipped:
                    training_progress.set_postfix(phase="backward", refresh=True)
                    result = training_client.forward_backward(
                        datums,
                        loss_fn="importance_sampling",
                    ).result()
                    training_progress.set_postfix(phase="optimizer", refresh=True)
                    training_client.optim_step(adam_params).result()
                    metrics.update(
                        {f"trainer/{key}": value for key, value in result.metrics.items()}
                    )

                # 汇总 rollout、搜索客户端和远程 trainer 指标。
                metrics.update(search_client.stats.metrics())

                # 按配置同时保存可续训 state 和可推理 sampler weights。
                # checkpoint 保存耗时也算在当前 step 的完整耗时内。
                if args.save_every > 0 and (step + 1) % args.save_every == 0:
                    training_progress.set_postfix(phase="checkpoint", refresh=True)
                    save_checkpoint(
                        training_client,
                        f"{args.run_name}-step-{step + 1}",
                    )

                # 记录完整 step 耗时，终端和 SwanLab 都能看到同一数值。
                step_seconds = perf_counter() - step_started
                metrics["time/step_seconds"] = step_seconds
                metrics["train/update_skipped"] = float(update_skipped)
                swanlab.log(metrics, step=step)

                loss_mean_text = (
                    "skipped" if update_skipped else f"{metrics['trainer/loss_mean']:.4f}"
                )
                training_progress.update(1)
                training_progress.set_postfix(
                    step_s=f"{step_seconds:.1f}",
                    loss_mean=loss_mean_text,
                    reward=f"{metrics['reward/mean']:.3f}",
                    refresh=True,
                )
                tqdm.write(
                    f"step={step + 1}/{args.max_steps} "
                    f"step_time={step_seconds:.1f}s "
                    f"loss_mean={loss_mean_text} "
                    f"mean_reward={metrics['reward/mean']:.3f} "
                    f"correct_rate={metrics['reward/correct']:.3f} "
                    f"mean_search_calls={metrics['rollout/search_calls']:.2f} "
                    f"input_tokens={int(metrics['train/tokens_per_rollout_batch'])} "
                    f"loss_tokens={int(metrics['train/loss_tokens_per_rollout_batch'])}"
                )

        # 无论周期保存频率如何，正常完成训练后始终保存一次最终 checkpoint。
        save_checkpoint(training_client, f"{args.run_name}-final")
    finally:
        # 即使训练中途抛出异常，也要结束 SwanLab run，避免实验一直显示为运行中。
        run.finish()


if __name__ == "__main__":
    main(parse_args())
