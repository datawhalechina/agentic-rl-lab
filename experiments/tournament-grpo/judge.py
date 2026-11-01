"""Tournament judge：作者的 OpenAI 兼容 chat 协议，以及本复现使用的 Jev（System One API）。

Adapted from PuffYang/Tournament_GRPO (eea826cf), Apache-2.0.

检查已保存的 judge 配置（仓库根目录）：
uv run python experiments/tournament-grpo/judge.py
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
MAX_ATTEMPTS = 3  # 每场比赛的请求次数上限（含首次）
# 作者 judge prompt 的两段原文，chat 和 Jev 共用。
JUDGE_ROLE = "You are an expert judge for deep-research quality.\n"
SELECTION_RULE = (
    "Selection rule:\n"
    "- Select exactly 1 winner(s).\n"
    "- Prefer responses that are accurate, complete, well-supported, and directly answer the query.\n"
)
# Jev 把比较对象放在 state、选择规则放在 instructions，所以不需要 JSON 输出要求。
JEV_INSTRUCTIONS = (
    "You are a strict deep-research tournament judge.\n" + JUDGE_ROLE
    + "Compare the candidate responses to the same query using the provided rubric.\n\n"
    + SELECTION_RULE.rstrip("\n")
)


def build_judge_context(query: str, rubrics: list[dict[str, str]], answers: list[str]) -> str:
    """按作者格式拼出题目、rubric 和候选答案。"""
    rubric_lines = [
        f"{i}. description: {item['description'].strip()}\n"
        f"   dimension: {item['dimension'].strip()}\n"
        f"   importance: {item['importance'].strip()}\n"
        f"   title: {item['title'].strip()}"
        for i, item in enumerate(rubrics, 1)
    ]
    rubric_block = "\n".join(rubric_lines) if rubric_lines else "No rubrics provided."
    candidates = "\n\n".join(f"Candidate {i}:\n{text}" for i, text in enumerate(answers, 1))
    return (f"Query:\n{query}\n\nRubric (description, dimension, importance, title):\n"
            f"{rubric_block}\n\nCandidates:\n{candidates}\n\n")


def build_chat_prompt(context: str) -> str:
    """作者的 chat judge prompt 原文。"""
    return (
        JUDGE_ROLE + "Compare the candidate responses to the same query using the rubric below.\n\n"
        + context + SELECTION_RULE
        + "- Return JSON only, without markdown or explanations.\n"
        + '- Use 1-based candidate numbers with this schema: {"winners": [1]}\n'
    )


def parse_winner(content: str, num_candidates: int) -> int:
    """沿用作者宽松的解析：优先读 JSON 里的 winners，否则取文本中的数字；候选编号从 1 开始。"""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(cleaned)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        winners = parsed.get("winners", parsed.get("winner", []))
    elif isinstance(parsed, list):
        winners = parsed
    elif isinstance(parsed, (int, float)):
        winners = [parsed]
    else:
        winners = re.findall(r"\d+", cleaned)
    if not isinstance(winners, list):
        winners = [winners]
    for value in winners:
        try:
            index = int(value) - 1
        except (ValueError, TypeError, OverflowError):
            continue
        if 0 <= index < num_candidates:
            return index
    raise ValueError(f"Tournament judge returned invalid winners: {cleaned}")


@dataclass
class JudgeClient:
    """每场比赛选 1 个胜者。"""

    api_url: str
    model: str
    api_key: str = field(default="", repr=False)
    backend: str = "chat"  # chat 是作者的接口；本复现用 jev
    long_context: bool = False
    seed: int = 0  # 作者用同一个 seed 控制 judge 采样和淘汰赛分组
    timeout: float = 240.0

    @classmethod
    def from_env(cls, env_path: str | Path) -> "JudgeClient":
        load_dotenv(env_path)
        backend = os.getenv("TOURNAMENT_JUDGE_BACKEND", "chat")
        jev = backend == "jev"
        judge = cls(
            api_url=os.getenv("TOURNAMENT_JUDGE_API_URL") or (JEV_API_URL if jev else ""),
            model=os.getenv("TOURNAMENT_JUDGE_MODEL_NAME") or ("jev-1.14.0" if jev else "Qwen2.5-72B-Instruct"),
            api_key=os.getenv("TOURNAMENT_JUDGE_API_KEY", ""),
            backend=backend,
            long_context=os.getenv("TOURNAMENT_JUDGE_LONG_CONTEXT", "false").lower() == "true",
        )
        # 启动前检查配置，避免采样完成后才发现 judge 不可用。
        if not judge.api_url or (jev and not judge.api_key):
            raise ValueError("judge 未配置：请在 experiments/tournament-grpo/.env 中填写 TOURNAMENT_JUDGE_*")
        return judge

    def judge(self, query: str, rubrics: list[dict[str, str]], answers: list[str]) -> tuple[int, Any]:
        """返回 (组内胜者下标, judge 原始输出)。

        作者在重试失败后随机选胜者；本实现直接抛出异常停止训练，避免用错误的奖励更新。
        """
        context = build_judge_context(query, rubrics, answers)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                if self.backend == "jev":
                    return self.ask_jev(context, len(answers))
                return self.ask_chat(build_chat_prompt(context), len(answers))
            except Exception as exc:
                error = self.describe_error(exc)
                if attempt < MAX_ATTEMPTS:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(
            f"Tournament judge failed after {MAX_ATTEMPTS} attempts; stopping training. error={error}"
        ) from None

    def describe_error(self, exc: Exception) -> str:
        """保留 HTTP 状态码和响应正文，并去掉其中可能出现的 Key。"""
        if isinstance(exc, httpx.HTTPStatusError):
            error = f"HTTP {exc.response.status_code}: {exc.response.text[:1500]}"
        else:
            error = str(exc)
        return error.replace(self.api_key, "[REDACTED]") if self.api_key else error

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.backend == "jev" and self.long_context:
            headers["TypeSafe-Danger-Zone-Long-Context"] = "true"
        return headers

    def ask_chat(self, prompt: str, num_candidates: int) -> tuple[int, str]:
        """作者的 chat 请求：temperature 0、max_tokens 1024。"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a strict deep-research tournament judge. Return only valid JSON without markdown."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0, "max_tokens": 1024, "top_p": 0.95, "seed": self.seed,
        }
        response = httpx.post(self.api_url, headers=self.headers(), json=payload, timeout=self.timeout, trust_env=False)
        response.raise_for_status()
        content = (response.json()["choices"][0]["message"].get("content") or "").strip()
        return parse_winner(content, num_candidates), content

    def ask_jev(self, context: str, num_candidates: int) -> tuple[int, dict[str, Any]]:
        """Jev 的 choice 问题直接返回一个候选编号，并附带 token usage。"""
        payload = {
            "model": self.model, "state": context,
            "questions": {"winner": {
                "type": "choice", "instructions": JEV_INSTRUCTIONS,
                "criteria": {str(i): f"Candidate {i}" for i in range(1, num_candidates + 1)},
            }},
        }
        response = httpx.post(self.api_url, headers=self.headers(), json=payload, timeout=self.timeout, trust_env=False)
        response.raise_for_status()
        body = response.json()
        answer = body["answers"]["winner"]
        index = int(answer["choice"]) - 1
        # 越界编号会被 Python 负下标悄悄接受，必须当作失败重试。
        if not 0 <= index < num_candidates:
            raise ValueError(f"Jev returned invalid choice: {answer['choice']}")
        return index, {"model": body["model"], "answer": answer, "usage": body.get("usage", {})}


def main() -> None:
    """用一道二选一的小题发一次真实请求，确认鉴权、模型和返回解析正常。"""
    judge = JudgeClient.from_env(Path(__file__).resolve().parent / ".env")
    winner, output = judge.judge("What is 2 + 2?", [], ["4", "5"])
    print(f"{judge.backend} / {judge.model} 调用成功，选中 Candidate {winner + 1}。")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
