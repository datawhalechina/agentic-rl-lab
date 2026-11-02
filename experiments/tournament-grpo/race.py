"""Deep Research Bench RACE：作者评分提示词、权重和相对参考报告的归一化。

Adapted from PuffYang/Tournament_GRPO (eea826cf), deep_research_bench/
ood_race_eval.py and utils/score_calculator.py. Prompt license: race_prompts/LICENSE.
"""

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv import load_dotenv


RACE_DIMS = ("comprehensiveness", "insight", "instruction_following", "readability")
PROMPT_DIR = Path(__file__).resolve().parent / "race_prompts"


class ThinkingEnabledError(RuntimeError):
    """The judge returned reasoning despite disabled thinking."""


def build_score_prompt(article: dict, reference: dict, criteria: dict) -> str:
    """作者原始中英文模板；裁判看细则描述，权重只在本地聚合时使用。"""
    template = (PROMPT_DIR / ("zh.txt" if article["language"] == "zh" else "en.txt")).read_text()
    descriptions = {dim: [{"criterion": c["criterion"], "explanation": c["explanation"]}
                          for c in items] for dim, items in criteria["criterions"].items()}
    return template.format(task_prompt=article["prompt"], article_1=article["article"],
                           article_2=reference["article"],
                           criteria_list=json.dumps(descriptions, ensure_ascii=False, indent=2))


def normalized_scores(output: dict, criteria: dict) -> dict[str, float]:
    """先按细则加权，再按维度加权；Overall 不是四项归一化分数的简单平均。"""
    totals = [0.0, 0.0]
    scores = {}
    for dim in RACE_DIMS:
        items = output[dim]
        if not items or len(items) != len(criteria["criterions"][dim]):
            raise ValueError(f"Incomplete RACE criteria: {dim}")
        weights = {c["criterion"]: c["weight"] for c in criteria["criterions"][dim]}
        sums, weight_sum = [0.0, 0.0], 0.0
        for item in items:
            label = item["criterion"].strip()
            # 作者的匹配顺序：原文、忽略大小写、子串，最后用该维度平均权重。
            weight = weights.get(label)
            if weight is None:
                weight = next((v for k, v in weights.items() if k.lower() == label.lower()), None)
            if weight is None:
                weight = next((v for k, v in weights.items()
                               if label.lower() in k.lower() or k.lower() in label.lower()), None)
            if weight is None:
                weight = sum(weights.values()) / len(weights)
            for i in range(2):
                value = float(item[f"article_{i + 1}_score"])
                if not math.isfinite(value) or not 0 <= value <= 10:
                    raise ValueError(f"Invalid RACE score: {dim}: {value}")
                sums[i] += value * weight
            weight_sum += weight
        averages = [value / weight_sum for value in sums]
        scores[dim] = averages[0] / sum(averages) if sum(averages) > 0 else 0.0
        for i in range(2):
            totals[i] += averages[i] * criteria["dimension_weight"][dim]
    scores["overall_score"] = totals[0] / sum(totals) if sum(totals) > 0 else 0.0
    return scores


@dataclass
class RaceJudge:
    """独立的 OpenAI 兼容评分接口，不使用 tournament 的胜者选择接口。"""

    api_url: str
    model: str
    api_key: str = field(default="", repr=False)
    timeout: float = 600.0
    max_tokens: int = 8192

    @classmethod
    def from_env(cls, path: Path):
        load_dotenv(path)
        url, model = os.getenv("RACE_JUDGE_API_URL", ""), os.getenv("RACE_JUDGE_MODEL_NAME", "")
        if not url or not model:
            raise ValueError("请配置 RACE_JUDGE_API_URL 和 RACE_JUDGE_MODEL_NAME；报告生成不需要裁判")
        return cls(url, model, os.getenv("RACE_JUDGE_API_KEY", ""))

    def score(self, article: dict, reference: dict, criteria: dict) -> dict:
        prompt = build_score_prompt(article, reference, criteria)
        responses = self.api_url.rstrip("/").endswith("/responses")
        if responses:
            payload = {"model": self.model, "input": prompt, "max_output_tokens": self.max_tokens,
                       "reasoning": {"effort": "none"}, "temperature": 0.0, "top_p": 0.95}
        else:
            payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "top_p": 0.95, "seed": 0, "max_tokens": self.max_tokens}
            if "deepseek" in self.model.lower():
                payload.update(thinking={"type": "disabled"}, reasoning_effort="none")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        for attempt in range(3):
            try:
                response = httpx.post(self.api_url, json=payload, headers=headers,
                                      timeout=self.timeout, trust_env=False)
                response.raise_for_status()
                body = response.json()
                if responses:
                    reasoning = [item for item in body.get("output", []) if item["type"] == "reasoning"]
                    thinking = any(part.get("text") for item in reasoning
                                   for part in item.get("summary", []) + item.get("content", []))
                    thinking |= bool(body.get("usage", {}).get("output_tokens_details", {}).get("reasoning_tokens", 0))
                    if thinking:
                        raise ThinkingEnabledError("RACE judge returned reasoning with thinking disabled")
                    if body.get("status") != "completed":
                        raise ValueError(f"RACE response incomplete: {body.get('incomplete_details')}")
                    text = "".join(part["text"] for item in body["output"] if item["type"] == "message"
                                   for part in item["content"] if part["type"] == "output_text")
                    finish_reason = body["status"]
                else:
                    choice = body["choices"][0]
                    if choice["message"].get("reasoning_content"):
                        raise ThinkingEnabledError("RACE judge returned reasoning with thinking disabled")
                    finish_reason = choice.get("finish_reason")
                    if finish_reason in {"length", "MAX_TOKENS"}:
                        raise ValueError("RACE judge output truncated; no score recorded")
                    text = choice["message"]["content"]
                fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
                content = fenced.group(1) if fenced else text[text.find("{"):text.rfind("}") + 1]
                parsed = json.loads(content)
                scores = normalized_scores(parsed, criteria)
                return {"id": article["id"], "prompt": article["prompt"], **scores,
                        "judge_output": parsed, "judge_raw_text": text,
                        "judge_model": body.get("model", self.model), "judge_usage": body.get("usage", {}),
                        "judge_finish_reason": finish_reason, "judge_thinking": "disabled"}
            except ThinkingEnabledError:
                raise
            except Exception as exc:
                error = str(exc).replace(self.api_key, "[REDACTED]") if self.api_key else str(exc)
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"RACE judge failed after 3 attempts: {error}") from None
