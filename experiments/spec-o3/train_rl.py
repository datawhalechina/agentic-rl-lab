r"""从 SFT state 权重开始，用多轮工具轨迹进行 GRPO 训练。

在 experiments/spec-o3/ 目录下运行，首次使用先执行 uv run trio login 和 uv run swanlab login：

uv run python train_rl.py --state-path '<PyTrio 权重控制台获取或训练终端打印的 state_path>'

uv run python train_rl.py \
    --state-path '<PyTrio 权重控制台获取或训练终端打印的 state_path>' \
    --epochs 1 \
    --batch-size 8 \
    --group-size 8 \
    --learning-rate 1e-6 \
    --save-every 50 \
    --swanlab-mode online
"""

import argparse
import asyncio
import json
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
import time

import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from protocol import build_rl_datum, encode_messages
from rollout import rollout


async def rollout_group(
    row,
    sampler,
    tokenizer,
    processor,
    output_dir,
    args,
    seed,
    progress,
):
    """共享第一轮输入，采样出一组开头，再并发完成每条独立轨迹。"""
    prompt, _, _ = await asyncio.to_thread(
        encode_messages,
        row["messages"],
        row["images"],
        tokenizer,
        processor,
        generation=True,
    )
    first_turn = await sampler.sample_async(
        prompt=prompt,
        num_samples=args.group_size,
        sampling_params=trio.SamplingParams(
            max_tokens=min(args.max_tokens, args.max_seq_len - len(prompt)),
            temperature=args.temperature,
            top_p=args.top_p,
            stop=[tokenizer.convert_tokens_to_ids("<|im_end|>")],
            seed=seed,
        ),
    )

    async def run_trajectory(index, sequence):
        trajectory = await rollout(
            row,
            sampler,
            tokenizer,
            processor,
            output_dir / str(index),
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            max_seq_len=args.max_seq_len,
            temperature=args.temperature,
            seed=seed + (index + 1) * args.max_turns,
            top_p=args.top_p,
            first_sequence=sequence,
        )
        # 每条完整轨迹结束就推进一次，同一 step 的所有 group 共用进度条。
        progress.update(1)
        return trajectory

    return await asyncio.gather(
        *(run_trajectory(index, sequence) for index, sequence in enumerate(first_turn.sequences))
    )


def trajectory_reward(trajectory, format_penalty):
    """奖励 = 答案正确得 1 分 − 格式错误罚分；不奖励工具调用次数。"""
    # 单独读取最终答案，允许“答案正确但缺少规定标签”得到部分分数。
    # 只识别思考结束后、回答开头的 YES/NO，不从思考或工具参数中猜答案。
    body = trajectory["turns"][-1]["text"].partition("</think>")[2].strip()
    answer = re.match(r"(?:<answer>\s*)?(?:\\boxed\{)?(YES|NO)\b", body)
    correct = answer is not None and answer[1] == trajectory["label"]
    reward = float(correct) - format_penalty * (not trajectory["format_ok"])
    return reward, correct


def build_training_samples(groups, format_penalty):
    """在完整 group 内计算 advantage，汇总当前 batch 的有效训练样本。"""
    samples = []
    rewards = []
    correct = []
    trajectories = []
    degenerate_groups = 0

    for group in groups:
        scores = [trajectory_reward(item, format_penalty) for item in group]
        group_rewards = np.asarray([score[0] for score in scores], dtype=np.float32)
        rewards.extend(group_rewards.tolist())
        correct.extend(score[1] for score in scores)
        trajectories.extend(group)

        # 同组奖励完全相同，没有相对学习信号；仍计入全部 rollout 指标。
        if np.ptp(group_rewards) == 0:
            degenerate_groups += 1
            continue

        # 沿用作者 VeRL 的样本标准差（ddof=1）及 GRPO 的 epsilon。
        advantages = (
            (group_rewards - group_rewards.mean())
            / (group_rewards.std(ddof=1) + 1e-6)
        )
        for trajectory, advantage in zip(group, advantages):
            tokens = sum(trajectory["action_mask"])
            # 没有生成动作的轨迹无法提供策略梯度，奖励仍参与组内归一化。
            if tokens:
                samples.append((build_rl_datum(trajectory, advantage), tokens))

    count = len(trajectories)
    metrics = {
        "train/reward": float(np.mean(rewards)),
        "train/accuracy": sum(correct) / count,
        "train/format_rate": sum(item["format_ok"] for item in trajectories) / count,
        "train/degenerate_fraction": degenerate_groups / len(groups),
        "train/trajectories": len(samples),
        "train/assistant_tokens": sum(tokens for _, tokens in samples),
        "rollout/mean_turns": sum(len(item["turns"]) for item in trajectories) / count,
        "rollout/mean_generated_tokens": (
            sum(sum(item["action_mask"]) for item in trajectories) / count
        ),
        "rollout/tool_calls": sum(
            "observation" in turn
            for item in trajectories
            for turn in item["turns"]
        ),
    }
    return samples, metrics


async def save_checkpoint(client, name):
    """分别保存可继续训练的 state 和可独立评测的 sampler 权重。"""
    state_future = await client.save_state_async(
        name=f"{name}-state",
    )
    sampler_future = await client.save_weights_for_sampler_async(
        name=f"{name}-sampler",
    )
    state, sampler = await asyncio.gather(state_future, sampler_future)
    tqdm.write(f"{name} state_path（继续训练）：{state.path}")
    tqdm.write(f"{name} sampler_path（推理评测）：{sampler.path}")


async def main(args):
    # 1. 加载 RL 训练集，从 state 恢复基模、LoRA 配置和训练权重。
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    service = trio.ServiceClient()
    if args.resume_optimizer:
        client = await service.create_training_client_from_state_with_optimizer_async(
            args.state_path,
        )
    else:
        client = await service.create_training_client_from_state_async(args.state_path)

    tokenizer = AutoTokenizer.from_pretrained(
        client.model_id,
        revision=args.model_revision,
    )
    processor = AutoImageProcessor.from_pretrained(
        client.model_id,
        revision=args.model_revision,
        backend="pil",
    )
    optimizer = trio.AdamParams(learning_rate=args.learning_rate)
    swanlab.init(
        project=args.project_name,
        experiment_name=args.run_name,
        config={
            **vars(args),
            "data": str(args.data),
            "base_model": client.model_id,
        },
        mode=args.swanlab_mode,
    )

    rng = random.Random(args.seed)
    step = 0
    updates = 0
    steps_per_epoch = (len(rows) + args.batch_size - 1) // args.batch_size
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(rows)
        with (
            tqdm(
                total=len(rows) * args.group_size,
                desc=f"RL epoch {epoch}",
                unit="traj",
                position=0,
            ) as progress,
            tqdm(total=0, unit="traj", position=1, leave=False) as step_progress,
        ):
            for start in range(0, len(rows), args.batch_size):
                started = time.monotonic()
                batch = rows[start : start + args.batch_size]
                step_in_epoch = start // args.batch_size + 1
                step_progress.reset(total=len(batch) * args.group_size)
                step_progress.set_description(f"Step {step_in_epoch}/{steps_per_epoch}")
                step_progress.set_postfix_str("准备采样")

                # 2. 当前 batch 共用一个固定版本的 sampler，完成全部组后才更新。
                sampler = await client.save_weights_and_get_sampling_client_async()
                step_progress.set_postfix_str("采样中")
                with TemporaryDirectory(prefix="spec-o3-rl-") as work_dir:
                    groups = await asyncio.gather(
                        *(
                            rollout_group(
                                row,
                                sampler,
                                tokenizer,
                                processor,
                                Path(work_dir) / str(index),
                                args,
                                seed=args.seed + (
                                    (step * args.batch_size + index)
                                    * (args.group_size + 1) * args.max_turns
                                ),
                                progress=step_progress,
                            )
                            for index, row in enumerate(batch)
                        )
                    )
                    samples, metrics = build_training_samples(groups, args.format_penalty)

                # Datum 内已经包含图片字节，后续更新不再依赖临时图片文件。
                if samples:
                    step_progress.set_postfix_str("更新参数")
                    data = [datum for datum, _ in samples]
                    tokens = sum(count for _, count in samples)

                    # 3. 按整个 batch 的 assistant token 数缩放 advantage。
                    for datum in data:
                        advantages = datum.loss_fn_inputs["advantages"].to_numpy()
                        datum.loss_fn_inputs["advantages"] = trio.types.TensorData.from_numpy(
                            advantages / tokens,
                        )

                    # 整批交给 PyTRIO 自动拆分、累积梯度，只更新一次参数。
                    backward = await client.forward_backward_async(
                        data,
                        loss_fn="ppo",
                        loss_fn_config={
                            "clip_low_threshold": 1 - args.clip_ratio,
                            "clip_high_threshold": 1 + args.clip_ratio,
                        },
                    )
                    update = await client.optim_step_async(optimizer)
                    result = await backward
                    await update
                    updates += 1
                    step_progress.set_postfix_str("完成")
                    # Future 返回整个 batch 聚合后的服务端指标。
                    metrics.update({
                        f"trainer/{key}": value
                        for key, value in result.metrics.items()
                    })
                else:
                    step_progress.set_postfix_str("无有效训练样本，跳过更新")

                # 4. 记录所有采样的表现，包括没有相对学习信号的 group。
                step += 1
                metrics["train/epoch"] = epoch
                metrics["train/updates"] = updates
                metrics["time/batch_seconds"] = time.monotonic() - started
                swanlab.log(metrics, step=step)
                progress.update(sum(len(group) for group in groups))
                progress.set_postfix(
                    reward=f"{metrics['train/reward']:.3f}",
                    accuracy=f"{metrics['train/accuracy']:.1%}",
                    updates=updates,
                )

                # 5. 按累计 rollout step 保存，此时本批的参数更新已完成。
                if step % args.save_every == 0:
                    step_progress.set_postfix_str("保存权重")
                    await save_checkpoint(client, f"{args.run_name}-step-{step}")
                    step_progress.set_postfix_str("已保存权重")

        # 当前 epoch 的全部参数更新完成后，分别保存两份权重。
        await save_checkpoint(client, f"{args.run_name}-epoch-{epoch}")

    swanlab.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 从已有权重开始训练；模型与 rank 由 state 决定。
    parser.add_argument(
        "--state-path",
        required=True,
        help="PyTrio 权重控制台获取或训练终端打印的 state_path",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="同时恢复 RL 优化器状态",
    )
    parser.add_argument("--model-revision", default="main")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "datasets/rl/bench_rl_train.jsonl",
    )

    # batch-size 为每批题目数，group-size 为每题采样的轨迹数。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--format-penalty", type=float, default=0.2)
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="每多少个累计 rollout step 保存 state 和 sampler，默认 100",
    )

    # 整条轨迹的轮数、上下文预算，以及每次生成的采样参数。
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--project-name", default="agentic-rl-lab-spec-o3")
    parser.add_argument("--run-name", default="spec-o3-rl")
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "disabled"],
        default="online",
    )
    asyncio.run(main(parser.parse_args()))
