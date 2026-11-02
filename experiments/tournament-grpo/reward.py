"""Tournament-GRPO 的二值格式奖励与淘汰赛奖励。

Adapted from PuffYang/Tournament_GRPO (eea826cf), Apache-2.0.
格式解析和淘汰赛流程沿用作者 tournament_grpo_reward.py，去掉了 verl 适配层。
"""

import random
import re
from dataclasses import dataclass
from typing import Any

from data import ResearchExample
from judge import JudgeClient


MAX_ROLLOUT_CHARS = 120000  # 作者 max_rollout_chars：送给 judge 的答案最多保留的字符数
TOURNAMENT_REPEATS = 4  # 整场淘汰赛重复次数 M；每次重新抽签
PLAYERS_PER_MATCH = 2  # 每场比赛人数 G；每场选 1 个胜者，胜者 +1 分

TAG_PATTERN = re.compile(r"<(/?)([A-Za-z_][A-Za-z0-9_]*)([^>]*)>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ATTRIBUTE_PATTERN = re.compile(r'\s+([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
ALLOWED_TAGS = {"think", "call_tool", "tool_output", "snippet", "webpage", "answer"}
TOOL_NAMES = {"google_search", "browse_webpage"}
# 标签层面的错误，按作者的固定顺序写入 failure_reasons。
TAG_ERRORS = ("tag:unknown_tags", "tag:invalid_attrs", "tag:broken_xml", "structure:stray_text_outside_tags")


# ---------------------------------------------------------------------------
# 格式奖励
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptBlock:
    """一个顶层标签块，例如 <think>...</think>；嵌套的子标签留在 content 里。"""

    tag: str
    attrs: dict[str, str]
    content: str


def parse_attrs(tag: str, text: str) -> tuple[dict[str, str], bool]:
    """解析标签属性；只有 call_tool 允许且必须带 name。"""
    attrs, cursor = {}, 0
    for match in ATTRIBUTE_PATTERN.finditer(text):
        key, value = match.groups()
        if text[cursor:match.start()].strip() or key in attrs:
            return attrs, False
        attrs[key] = value
        cursor = match.end()
    allowed = {"name"} if tag == "call_tool" else set()
    valid = not text[cursor:].strip() and not (attrs.keys() - allowed)
    return attrs, valid and (tag != "call_tool" or "name" in attrs)


def parse_transcript(text: str) -> tuple[list[TranscriptBlock], set[str]]:
    """把完整轨迹切成顶层标签块，并收集标签错误。

    沿用作者的栈式恢复：observation 里的标签同样参与解析；关闭标签与栈顶不匹配时，
    回退到栈里最近的同名标签，中间未闭合的标签直接丢弃。
    """
    blocks, stack, errors = [], [], set()
    cursor = 0  # 上一个标签的结束位置，用来发现标签外的杂文
    for match in TAG_PATTERN.finditer(text):
        if not stack and text[cursor:match.start()].strip():
            errors.add("structure:stray_text_outside_tags")
        closing, tag, attributes = match.groups()
        cursor = match.end()
        if tag not in ALLOWED_TAGS:
            errors.add("tag:unknown_tags")

        if not closing:
            attrs, valid = parse_attrs(tag, attributes)
            if not valid:
                errors.add("tag:invalid_attrs")
            stack.append((tag, attrs, match.end()))
            continue

        if attributes.strip():
            errors.add("tag:invalid_attrs")
        if stack and stack[-1][0] == tag:
            opened = stack.pop()
        else:
            # 栈顶不匹配：回退到最近的同名开标签，中间未闭合的标签一起丢弃。
            errors.add("tag:broken_xml")
            index = max((i for i, (open_tag, _, _) in enumerate(stack) if open_tag == tag), default=None)
            if index is None:
                continue
            opened, stack = stack[index], stack[:index]
        if not stack:
            open_tag, attrs, start = opened
            blocks.append(TranscriptBlock(open_tag, attrs, text[start:match.start()]))

    if not stack and text[cursor:].strip():
        errors.add("structure:stray_text_outside_tags")
    if stack:
        errors.add("tag:broken_xml")
    return blocks, errors


def extract_answer(text: str) -> str:
    """只接受唯一、非空、不含标签且位于末尾的 <answer>。"""
    matches = list(ANSWER_PATTERN.finditer(text))
    if len(matches) != 1:
        return ""
    answer = matches[0].group(1).strip()
    if not answer or TAG_PATTERN.search(answer) or text[matches[0].end():].strip():
        return ""
    return answer


def tool_name(block: TranscriptBlock) -> str:
    return block.attrs.get("name", "").strip()


def is_tool_cycle(blocks: list[TranscriptBlock], index: int, name: str) -> bool:
    """blocks[index:index+3] 是否是一轮非空的 call_tool(name) → tool_output → think。"""
    if index + 2 >= len(blocks):
        return False
    call, observation, think = blocks[index:index + 3]
    return (call.tag == "call_tool" and tool_name(call) == name and bool(call.content.strip())
            and observation.tag == "tool_output" and bool(observation.content.strip())
            and think.tag == "think" and bool(think.content.strip()))


def check_structure(blocks: list[TranscriptBlock]) -> str | None:
    """作者的轨迹结构：think → 若干研究段 → answer；返回第一条失败原因，合法时返回 None。

    每个研究段 = 至少 1 轮 google_search 循环 + 任意轮 browse_webpage 循环。
    只在标签无错误且 extract_answer 成功时调用；此时唯一的 answer 必然是最后一个顶层块，
    所以作者的 must_end_with_answer、invalid_tool_cycle_order 两条检查不会触发，这里省略。
    """
    if blocks[0].tag != "think" or not blocks[0].content.strip():
        return "structure:must_start_with_non_empty_think"
    if len(blocks) <= 2:
        return "structure:must_have_at_least_one_tool_cycle"
    index = 1
    while index < len(blocks) - 1:
        if not is_tool_cycle(blocks, index, "google_search"):
            return "structure:each_segment_must_start_with_google_search_cycle"
        while is_tool_cycle(blocks, index, "google_search"):
            index += 3
        while is_tool_cycle(blocks, index, "browse_webpage"):
            index += 3
    return None


def analyze_format(text: str) -> dict[str, Any]:
    """格式奖励为 0 或 1：没有任何失败原因时得 1。同时取出送给 judge 的最终答案。"""
    blocks, errors = parse_transcript(text)
    failures = [reason for reason in TAG_ERRORS if reason in errors]
    if any(block.tag == "call_tool" and tool_name(block) not in TOOL_NAMES for block in blocks):
        failures.append("tool:unknown_tool_name")
    answer = extract_answer(text)
    if not answer:
        failures.append("answer:invalid_or_missing_final_answer")
    if not failures:
        structure_failure = check_structure(blocks)
        if structure_failure:
            failures.append(structure_failure)
    return {"format_reward": float(not failures), "failure_reasons": failures, "rollout_answer": answer}


# ---------------------------------------------------------------------------
# 淘汰赛奖励
# ---------------------------------------------------------------------------


def normalize_scores(scores: list[float]) -> list[float]:
    """组内 min-max 归一化；全部同分时每个候选记 0.5。"""
    low, high = min(scores), max(scores)
    return [0.5] * len(scores) if high == low else [(value - low) / (high - low) for value in scores]


def run_tournament(
    query: str, rubrics: list[dict[str, str]], answers: list[str], judge: JudgeClient,
) -> dict[str, Any]:
    """作者默认设置：每组 2 人、每组 1 个胜者、整场重复 4 次、每胜一场 +1 分。

    K=8 时每次 4 + 2 + 1 场比赛，4 次共 28 场。
    """
    scores = [0.0] * len(answers)
    matches = []
    rng = random.Random(judge.seed)
    for repeat in range(1, TOURNAMENT_REPEATS + 1):
        candidates = list(range(len(answers)))
        round_index = 0
        while len(candidates) > 1:
            round_index += 1
            rng.shuffle(candidates)
            # 作者按轮转方式分组：第 i 个候选进第 i % num_groups 组，人数为奇数时会出现 3 人组。
            num_groups = len(candidates) // PLAYERS_PER_MATCH
            groups = [candidates[i::num_groups] for i in range(num_groups)]
            winners = []
            for group in groups:
                local_index, output = judge.judge(query, rubrics, [answers[i] for i in group])
                winner = group[local_index]
                scores[winner] += 1.0
                winners.append(winner)
                # candidates 是 answers 的下标，可以据此还原每场比赛的 judge 输入。
                matches.append({"repeat": repeat, "round": round_index, "candidates": group,
                                "winner": winner, "output": output})
            candidates = winners
    return {"raw_scores": scores, "rewards": normalize_scores(scores), "matches": matches}


def score_group(
    example: ResearchExample, solutions: list[str], judge: JudgeClient,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """给同一道题的一整组轨迹打分：总奖励 = 淘汰赛奖励 + 格式奖励（0–2）。

    judge 只看提取出的最终答案。没有答案的候选仍以空字符串参赛，归一化之后再把淘汰赛奖励置 0。
    返回每个候选的奖励明细和本组的 judge 日志。
    """
    formats = [analyze_format(text) for text in solutions]
    answers = [info["rollout_answer"][:MAX_ROLLOUT_CHARS] for info in formats]
    tournament = run_tournament(example.query.strip(), example.rubrics, answers, judge)
    results = []
    for info, answer, reward, raw in zip(formats, answers, tournament["rewards"], tournament["raw_scores"], strict=True):
        reward = reward if answer else 0.0
        results.append({**info, "rollout_answer": answer, "score": reward + info["format_reward"],
                        "tournament_reward": reward, "tournament_raw_score": raw})
    log = {
        "id": example.id, "judge_backend": judge.backend, "judge_model": judge.model,
        "judge_long_context": judge.long_context, "answers": answers,
        "raw_scores": tournament["raw_scores"], "tournament_rewards": [r["tournament_reward"] for r in results],
        "format_rewards": [r["format_reward"] for r in results], "matches": tournament["matches"],
    }
    return results, log
