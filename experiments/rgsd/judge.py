"""Rubric judge：对一条回答逐条判定 rubric 是否满足，并给出加权满足率。

Judge prompt 逐字取自 RGSD 论文附录 C.2（Figure 12）：一次调用返回全部 K 条判定，
得分为权重归一化后的满足比例（论文公式 1）。

提供两种后端：

- ``trio``（默认）：用 PyTRIO 上的基础模型当 judge，不需要额外的 API key。
- ``chat``：任意 OpenAI 兼容的 ``/chat/completions`` 接口，配置写在
  ``experiments/rgsd/.env``（见 ``.env.example``）。

论文评测用 gpt-5.4、条件增益诊断用 gpt-5.4-mini；本复现默认的 PyTRIO judge 更弱，
所以绝对分数不能和论文对比，只看同一 judge 下两种条件之间的差值。
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
import pytrio as trio

from sampling import stop_strings


SCRIPT_DIR = Path(__file__).resolve().parent
MAX_ATTEMPTS = 3  # 每条回答的 judge 请求次数上限（含首次）

JUDGE_TEMPLATE = (
    "You are an expert evaluator. Given a user question, a candidate response, and "
    "a list of evaluation criteria, decide for each criterion whether the response "
    "satisfies it.\n\n"
    "=== Question ===\n{question}\n\n"
    "=== Response ===\n{response}\n\n"
    "=== Criteria ===\n{criteria}\n\n"
    'For each criterion, output a JSON object {{"id": <int>, "satisfied": '
    '<true|false>, "reason": <one sentence>}}. Output nothing other than a JSON '
    "array of these objects."
)


def build_judge_prompt(question: str, response: str, rubrics: list[dict[str, Any]]) -> str:
    """按论文模板拼出 judge 的 user 消息。"""
    criteria = "\n".join(
        f"{index}. {item['criterion'].strip()}" for index, item in enumerate(rubrics, 1)
    )
    return JUDGE_TEMPLATE.format(
        question=question.strip(), response=response.strip(), criteria=criteria
    )


def parse_verdicts(content: str, num_criteria: int) -> list[bool]:
    """从 judge 输出里解析出 K 个布尔判定；缺条目或格式错误时抛 ValueError。"""
    # judge 即使关了 thinking 也可能带上思考块或 markdown 代码围栏，先剥掉。
    cleaned = content.rpartition("</think>")[2].strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("judge 输出中没有 JSON 数组")
    items = json.loads(cleaned[start : end + 1])
    if not isinstance(items, list):
        raise ValueError("judge 输出不是 JSON 数组")

    verdicts: dict[int, bool] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("satisfied"), bool):
            raise ValueError(f"judge 条目格式错误: {item!r}")
        verdicts[int(item["id"])] = item["satisfied"]
    # 必须恰好覆盖 1..K；少判或多判都说明这次输出不可信，交给上层重试。
    if sorted(verdicts) != list(range(1, num_criteria + 1)):
        raise ValueError(
            f"judge 应返回 {num_criteria} 条判定，实际 id 为 {sorted(verdicts)}"
        )
    return [verdicts[index] for index in range(1, num_criteria + 1)]


def rubric_score(verdicts: list[bool], rubrics: list[dict[str, Any]]) -> float:
    """论文公式 1：满足条目的权重之和 / 全部权重之和。"""
    total = sum(item["weight"] for item in rubrics)
    met = sum(item["weight"] for item, ok in zip(rubrics, verdicts, strict=True) if ok)
    return met / total


@dataclass(frozen=True)
class JudgeResult:
    score: float
    verdicts: list[bool]
    input_tokens: int
    output_tokens: int
    attempts: int


@dataclass
class TrioJudge:
    """用 PyTRIO 基础模型当 judge。``complete`` 返回 (judge 原始文本, 输入 token 数, 输出 token 数)。"""

    sampling_client: Any
    tokenizer: Any
    name: str
    max_tokens: int = 4096

    @classmethod
    async def create(
        cls, service_client: Any, base_model: str, max_tokens: int
    ) -> "TrioJudge":
        client = await service_client.create_sampling_client_async(base_model=base_model)
        return cls(client, client.get_tokenizer(), f"trio:{base_model}", max_tokens)

    async def complete(self, prompt: str) -> tuple[str, int, int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        # 判定需要确定性：temperature 0，重试靠的是解析失败而不是换随机种子。
        params = trio.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=0.0,
            stop=stop_strings(self.tokenizer),
        )
        response = await self.sampling_client.sample_async(
            prompt=trio.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=params,
            return_text=False,
        )
        tokens = list(response.sequences[0].tokens)
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text, len(prompt_ids), len(tokens)


@dataclass
class ChatJudge:
    """OpenAI 兼容 chat 接口的 judge，接口与 TrioJudge 相同。"""

    api_url: str
    model: str
    api_key: str = field(default="", repr=False)
    temperature: float | None = 0.0
    timeout: float = 240.0
    name: str = ""

    def __post_init__(self) -> None:
        self.name = f"chat:{self.model}"
        self._client = httpx.AsyncClient(timeout=self.timeout)

    @classmethod
    def from_env(cls, env_path: Path) -> "ChatJudge":
        load_dotenv(env_path)
        api_url = os.getenv("RGSD_JUDGE_API_URL", "")
        model = os.getenv("RGSD_JUDGE_MODEL_NAME", "")
        if not api_url or not model:
            raise ValueError(
                f"chat judge 未配置：请在 {env_path} 中填写 RGSD_JUDGE_API_URL 和 "
                "RGSD_JUDGE_MODEL_NAME"
            )
        # 部分模型（如 gpt-5 系列）只接受默认 temperature，留空表示不传该字段。
        raw_temperature = os.getenv("RGSD_JUDGE_TEMPERATURE", "0")
        return cls(
            api_url=api_url,
            model=model,
            api_key=os.getenv("RGSD_JUDGE_API_KEY", ""),
            temperature=float(raw_temperature) if raw_temperature else None,
        )

    async def complete(self, prompt: str) -> tuple[str, int, int]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = await self._client.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        return (
            body["choices"][0]["message"]["content"] or "",
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )


async def judge_response(
    backend: TrioJudge | ChatJudge,
    question: str,
    response: str,
    rubrics: list[dict[str, Any]],
    semaphore: asyncio.Semaphore,
) -> JudgeResult | None:
    """对一条回答打分；连续 MAX_ATTEMPTS 次失败返回 None，由调用方决定如何处理。"""
    # 空回答不必调用 judge：没有内容就不可能满足任何条目。
    if not response.strip():
        return JudgeResult(0.0, [False] * len(rubrics), 0, 0, 0)

    prompt = build_judge_prompt(question, response, rubrics)
    input_tokens = output_tokens = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with semaphore:
                content, used_in, used_out = await backend.complete(prompt)
            input_tokens += used_in
            output_tokens += used_out
            verdicts = parse_verdicts(content, len(rubrics))
        except (ValueError, KeyError, TypeError, httpx.HTTPError):
            # 解析失败和网络错误都重试；其余异常（如认证失败）直接向上抛出。
            await asyncio.sleep(1.5 * attempt)
            continue
        return JudgeResult(
            rubric_score(verdicts, rubrics), verdicts, input_tokens, output_tokens, attempt
        )
    return None


async def build_judge(
    backend: str, service_client: Any, trio_model: str, max_tokens: int
) -> TrioJudge | ChatJudge:
    """按命令行选择创建 judge 后端。"""
    if backend == "trio":
        return await TrioJudge.create(service_client, trio_model, max_tokens)
    if backend == "chat":
        return ChatJudge.from_env(SCRIPT_DIR / ".env")
    raise ValueError(f"未知的 judge 后端: {backend}")


if __name__ == "__main__":
    # 不发任何请求的本地自检：uv run python experiments/rgsd/judge.py
    demo_rubrics = [{"criterion": "States beriberi.", "weight": 10},
                    {"criterion": "Mentions thiamine.", "weight": 5}]
    demo_output = (
        '```json\n[{"id": 1, "satisfied": true, "reason": "ok"},\n'
        ' {"id": 2, "satisfied": false, "reason": "missing"}]\n```'
    )
    parsed = parse_verdicts(demo_output, 2)
    assert parsed == [True, False], parsed
    assert abs(rubric_score(parsed, demo_rubrics) - 10 / 15) < 1e-9
    try:
        parse_verdicts('[{"id": 1, "satisfied": true}]', 2)
    except ValueError:
        pass
    else:
        raise AssertionError("缺条目时应当报错")
    print("judge self-test passed")
