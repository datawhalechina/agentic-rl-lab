"""共用的多轮光谱审核循环：采样 → 绘图 → 回填观察 → 再采样。"""

from copy import deepcopy

import numpy as np
import pytrio as trio

from protocol import encode_messages, parse_action, tool_continuation
from tools import spectral_visualization_tool


async def rollout(
    row,
    sampler,
    tokenizer,
    image_processor,
    output_dir,
    max_turns=8,
    max_tokens=2048,
    max_seq_len=16384,
    temperature=0.6,
    seed=42,
    top_p=0.95,
    first_sequence=None,
):
    """执行一条光谱审核轨迹，保留每轮生成结果和用于后续 RL 的 token 信息。"""
    # 1. 构造初始图文输入，并加载可供工具重绘的原始光谱。
    output_dir.mkdir(parents=True, exist_ok=True)
    messages = deepcopy(row["messages"])
    prompt, _, _ = encode_messages(
        messages,
        row["images"],
        tokenizer,
        image_processor,
        generation=True,
    )
    with np.load(row["spectrum"]) as spectrum:
        wavelength, flux, redshift = (
            spectrum["wavelength"],
            spectrum["flux"],
            float(spectrum["redshift"]),
        )

    # 初始 prompt 只提供上下文；生成 token 和工具观察在下面逐轮追加。
    target_tokens = [0] * len(prompt)
    logprobs = [0.0] * len(prompt)
    action_mask = [0.0] * len(prompt)
    turns = []
    prediction = None
    finish_reason = "max_turns"
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    for turn in range(max_turns):
        # 2. 在剩余上下文预算内，生成下一段思考和动作。
        remaining = max_seq_len - len(prompt)
        if remaining <= 0:
            finish_reason = "max_seq_len"
            break

        # GRPO 第一轮已按 num_samples 成组采样；后续沿各自的上下文继续。
        if turn == 0 and first_sequence is not None:
            sequence = first_sequence
        else:
            response = await sampler.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=trio.SamplingParams(
                    max_tokens=min(max_tokens, remaining),
                    temperature=temperature,
                    top_p=top_p,
                    stop=[im_end],
                    seed=seed + turn,
                ),
            )
            sequence = response.sequences[0]
        text = tokenizer.decode(sequence.tokens, skip_special_tokens=False)
        action = parse_action(text)

        # 追加服务实际生成的 token，不重新编码 assistant 历史。
        if sequence.tokens:
            prompt = trio.ModelInput(chunks=[
                *prompt.chunks,
                trio.types.EncodedTextChunk(tokens=sequence.tokens),
            ])
        target_tokens.extend(sequence.tokens)
        logprobs.extend(sequence.logprobs)
        action_mask.extend([1.0] * len(sequence.tokens))
        record = {
            "turn": turn,
            "text": text,
            "tokens": sequence.tokens,
            "logprobs": sequence.logprobs,
            "stop_reason": sequence.stop_reason,
            "action": action,
        }
        turns.append(record)

        # 3. 最终答案、非法动作和轮数耗尽都会结束当前轨迹。
        if action["kind"] == "answer":
            prediction = action["prediction"]
            finish_reason = "answer"
            break
        if action["kind"] == "invalid":
            finish_reason = "invalid_action"
            break
        if turn + 1 == max_turns:
            break

        # 4. 按模型选择的波段绘图，再追加 assistant 消息和工具观察。
        observation = spectral_visualization_tool(
            wavelength,
            flux,
            redshift,
            action["wavelength_range"],
            action["label"],
            output_dir / f"turn-{turn}.png",
        )
        record["observation"] = observation
        reasoning, _, content = text.partition("</think>")
        messages.append({
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": content.removesuffix("<|im_end|>").strip(),
            "tool_calls": [],
        })
        messages.append({"role": "tool", "content": observation["content"]})

        # 5. 只编码新增的观察和下一轮 assistant 前缀。
        continuation = tool_continuation(
            messages,
            observation["images"],
            tokenizer,
            image_processor,
        )
        # stop token 若被 sampler 排除，补到上下文；它没有采样 logprob，也不参与 loss。
        separator = "\n" if sequence.tokens[-1] == im_end else "<|im_end|>\n"
        context = trio.ModelInput(chunks=[
            trio.types.EncodedTextChunk(
                tokens=tokenizer.encode(separator, add_special_tokens=False),
            ),
            *continuation.chunks,
        ])
        record["observation_delivered"] = len(prompt) + len(context) < max_seq_len
        if not record["observation_delivered"]:
            finish_reason = "max_seq_len"
            break

        # 工具观察属于上下文，target、logprob 和动作 mask 都填零。
        prompt = trio.ModelInput(chunks=[*prompt.chunks, *context.chunks])
        target_tokens.extend([0] * len(context))
        logprobs.extend([0.0] * len(context))
        action_mask.extend([0.0] * len(context))

    return {
        "id": row["id"],
        "task": row["task"],
        "label": row["label"],
        "prediction": prediction,
        "format_ok": finish_reason == "answer",
        "finish_reason": finish_reason,
        "turns": turns,
        "initial_messages": row["messages"],
        "initial_images": row["images"],
        "model_input": prompt,
        "target_tokens": target_tokens,
        "logprobs": logprobs,
        "action_mask": action_mask,
    }
