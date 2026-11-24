"""计算 Search-R1 的格式奖励和答案精确匹配奖励。"""

import re
import unicodedata
from dataclasses import dataclass


ANSWER_PATTERN = re.compile(r"\s*Answer:(.*)", re.IGNORECASE)
ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RewardResult:
    """保存最终 reward 及其判定细节。"""

    reward: float
    valid_format: bool
    exact_match: bool
    answer: str | None


def normalize_answer(text: str) -> str:
    """统一答案格式：转小写、去除标点和英文冠词，并合并多余空格。"""
    lowered = text.lower()
    without_punctuation = "".join(
        char for char in lowered if not unicodedata.category(char).startswith("P")
    )
    without_articles = ARTICLE_PATTERN.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def extract_answer(text: str) -> str | None:
    """提取唯一 Answer: 标签行上的非空答案。"""
    matches = [
        match.group(1)
        for line in text.splitlines()
        if (match := ANSWER_PATTERN.fullmatch(line))
    ]
    if len(matches) != 1:
        return None
    answer = matches[0].strip()
    return answer or None


def score_answer(text: str, references: list[str]) -> RewardResult:
    """按 1.0、0.0、-0.1 三档规则计算轨迹奖励。"""
    answer = extract_answer(text)
    if answer is None:
        return RewardResult(-0.1, False, False, None)
    normalized = normalize_answer(answer)
    exact_match = any(normalized == normalize_answer(reference) for reference in references)
    return RewardResult(float(exact_match), True, exact_match, answer)
