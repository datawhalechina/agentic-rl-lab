"""生成式 critic：把 macro-step 边界状态的价值估计建模为生成任务。

critic 与 actor 共享同一份采样权重，仅靠 prompt、上下文和奖励区分角色。
它可以看到 actor 看不到的特权信息（专家 walkthrough），先推理再在结尾
给出 <value>0.42</value> 形式的数值估计。
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytrio as trio

from protocol import stop_sequences


CRITIC_SYSTEM_PROMPT = """You are a value estimator for an autonomous agent in the text-only ALFWorld environment.
The agent is mid-episode. You see its task, its full interaction history, and privileged
reference information (an expert walkthrough) that the agent itself cannot see.

Estimate the probability that this agent will eventually complete the task from the
current state. Judge:
- what the agent has already accomplished and understood so far,
- whether its current hypothesis and plan still lead to a valid solution (check it
  against the walkthrough),
- what obstacles remain.

Be concise. End your reply with exactly one line of the form
<value>0.42</value>
where the number is your probability estimate between 0.00 and 1.00. Do not call tools."""

VALUE_PATTERN = re.compile(
    r"<value>\s*([0-9]*\.?[0-9]+)\s*</value>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CriticRequest:
    """一个状态的 K 次独立估值请求。"""

    prompt_tokens: list[int]
    num_samples: int
    seed: int


@dataclass(frozen=True)
class CriticSample:
    """一次估值生成及其解析结果。"""

    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    text: str
    value: float | None


def parse_value(text: str) -> float | None:
    """取最后一个 <value>...</value>，收敛到 [0, 1]；没有则返回 None。"""
    matches = VALUE_PATTERN.findall(text)
    if not matches:
        return None
    value = float(matches[-1])
    if not math.isfinite(value):
        return None
    return min(1.0, max(0.0, value))


def condensed_history(messages: Sequence[dict[str, Any]]) -> str:
    """把 actor 的 chat 历史压成 critic 可读的紧凑文本。

    tool observation 里的 Available actions 列表只服务 actor 选动作，
    对估值没有信息量，直接裁掉以控制 critic prompt 长度。
    """
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role == "system":
            continue
        if role == "user":
            lines.append(f"Initial situation:\n{content}")
        elif role == "assistant":
            lines.append(f"Actor: {content}")
        elif role == "tool":
            cut = content.find("Available actions")
            if cut >= 0:
                content = content[:cut].rstrip()
            lines.append(f"Environment: {content}")
    return "\n\n".join(lines)


def critic_messages(
    *,
    task: str,
    walkthrough: Sequence[str],
    history: str,
) -> list[dict[str, Any]]:
    """构造 critic 的一次估值 prompt。"""
    walkthrough_lines = [
        f"  {index + 1}. {step}" for index, step in enumerate(walkthrough)
    ] or ["  (empty)"]
    user_lines = [
        "An ALFWorld episode is in progress.",
        f"Task: {task}",
        "",
        "Privileged reference (expert walkthrough, hidden from the agent):",
        *walkthrough_lines,
        "",
        "Interaction history:",
        history if history else "(The episode has just started.)",
        "",
        "Estimate the probability that the agent eventually completes the task "
        "from the current state.",
    ]
    return [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def _one_dimensional_tokens(value: Any) -> list[int]:
    # 与 protocol.py 相同：必须用 Mapping 而不是 dict，
    # 否则接不住 transformers 的 BatchEncoding（UserDict 子类）。
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def render_critic_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
) -> list[int]:
    """渲染 critic prompt；与 actor 不同，不携带工具定义。"""
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return _one_dimensional_tokens(rendered)


def critic_prompt_for_state(
    tokenizer: Any,
    state_messages: Sequence[dict[str, Any]],
    *,
    task: str,
    walkthrough: Sequence[str],
) -> list[int]:
    """从 actor 侧状态一步得到 critic prompt token。"""
    return render_critic_prompt(
        tokenizer,
        critic_messages(
            task=task,
            walkthrough=walkthrough,
            history=condensed_history(state_messages),
        ),
    )


async def _sample_async(
    sampling_client: Any,
    request: CriticRequest,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str],
) -> Any:
    return await sampling_client.sample_async(
        prompt=trio.ModelInput.from_ints(request.prompt_tokens),
        num_samples=request.num_samples,
        sampling_params=trio.SamplingParams(
            max_tokens=max_tokens,
            seed=request.seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        ),
        return_text=True,
    )


def sample_critic_values(
    sampling_client: Any,
    tokenizer: Any,
    requests: Sequence[CriticRequest],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[list[CriticSample]]:
    """并发采样所有请求；每个请求返回 num_samples 个估值样本。"""
    if not requests:
        return []
    stop = stop_sequences(tokenizer)

    async def run() -> list[Any]:
        calls = [
            _sample_async(
                sampling_client,
                request,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            for request in requests
        ]
        return list(await asyncio.gather(*calls))

    responses = asyncio.run(run())
    batches: list[list[CriticSample]] = []
    for request, response in zip(requests, responses, strict=True):
        sequences = list(response.sequences)
        if len(sequences) != request.num_samples:
            raise ValueError(
                f"critic 采样数量 {len(sequences)} != 请求数量 {request.num_samples}"
            )
        samples: list[CriticSample] = []
        for sequence in sequences:
            tokens = [int(token) for token in sequence.tokens]
            logprobs = [float(value) for value in sequence.logprobs]
            if len(tokens) != len(logprobs):
                raise ValueError("critic completion token 与 old logprob 长度不一致")
            text = sequence.text
            if text is None:
                text = tokenizer.decode(tokens, skip_special_tokens=True)
            text = str(text)
            samples.append(
                CriticSample(
                    prompt_tokens=request.prompt_tokens,
                    completion_tokens=tokens,
                    logprobs=logprobs,
                    text=text,
                    value=parse_value(text),
                )
            )
        batches.append(samples)
    return batches


def endpoint_value(samples: Sequence[CriticSample]) -> tuple[float, int]:
    """终点估值：解析成功的样本取均值；全部失败时悲观取 0.0。

    返回 (估值, 解析失败数)。
    """
    values = [sample.value for sample in samples if sample.value is not None]
    failures = len(samples) - len(values)
    if not values:
        return 0.0, failures
    return sum(values) / len(values), failures
