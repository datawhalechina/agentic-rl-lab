"""训练、诊断和评测共用的 PyTRIO 采样工具。"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

import pytrio as trio

from prompts import mentions_rubric, render_chat_prompt, split_thinking


def stop_strings(tokenizer: Any) -> list[str]:
    """模型的结束符；Qwen 的 eos 就是 ``<|im_end|>``，去重后只保留一个。"""
    return list(
        dict.fromkeys(token for token in (tokenizer.eos_token, "<|im_end|>") if token)
    )


def build_sampling_params(
    tokenizer: Any,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int = -1,
    seed: int | None = None,
) -> trio.SamplingParams:
    """统一构造采样参数，保证各脚本的停止条件一致。"""
    return trio.SamplingParams(
        max_tokens=max_tokens,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=stop_strings(tokenizer),
    )


def decode_completion(tokenizer: Any, tokens: list[int]) -> str:
    """解码 completion 并去掉结束符。

    保留 ``</think>`` 这类普通特殊 token，才能可靠地切出最终回答，因此不能直接用
    ``skip_special_tokens=True``。
    """
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    for token in stop_strings(tokenizer):
        text = text.replace(token, "")
    return text


def describe_answer(text: str, tokenizer: Any, enable_thinking: bool) -> dict[str, Any]:
    """把一段生成文本拆成 thinking / answer，并记录长度与 rubric 泄漏标记。"""
    thinking, answer = split_thinking(text, enable_thinking)
    return {
        "answer": answer,
        "thinking": thinking,
        "answer_tokens": len(tokenizer.encode(answer, add_special_tokens=False)),
        "answer_mentions_rubric": mentions_rubric(answer),
        "thinking_mentions_rubric": mentions_rubric(thinking),
    }


async def generate_answer(
    sampler: Any,
    tokenizer: Any,
    message: str,
    *,
    enable_thinking: bool,
    params: trio.SamplingParams,
    semaphore: asyncio.Semaphore,
    usage: Counter | None = None,
) -> dict[str, Any]:
    """对单条 user 消息生成一个回答，并返回拆分后的最终答案与统计信息。"""
    prompt_ids = render_chat_prompt(tokenizer, message, enable_thinking)
    async with semaphore:
        response = await sampler.sample_async(
            prompt=trio.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=params,
            return_text=False,
        )
    sequence = response.sequences[0]
    if usage is not None:
        usage["gen_input_tokens"] += len(prompt_ids)
        usage["gen_output_tokens"] += len(sequence.tokens)

    text = decode_completion(tokenizer, list(sequence.tokens))
    return {
        **describe_answer(text, tokenizer, enable_thinking),
        "completion_tokens": len(sequence.tokens),
        "stop_reason": sequence.stop_reason,
    }
