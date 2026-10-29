"""判别决策格式编译：canonical record → chat messages → tokens/weights。

- 候选符号表为 62 个单 token 符号（A-Z a-z 0-9），choice 按插入序、score 按等级序、
  noul 固定 [no, yes]；
- assistant 内容是 JSON 骨架，值的位置放占位符 marker。不加 special token（TRIO 服务端
  词表不可扩展），用已有单 token "decision"（Qwen3.5 tokenizer id 61781）；
- labels 只落在 marker 位置（右移后 = marker 前一位置预测答案符号），其余全 0。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytrio as trio

SYSTEM_PROMPT = (
    "You are a careful decision assistant. Use the state and decision schema in "
    "the user message to make the requested decisions. For every field, choose "
    "exactly one answer symbol (e.g. A, B, C, ...) from its listed options and "
    "return one valid JSON object mapping each field name to its chosen symbol. "
    "Use the field names and symbols exactly as given. Do not include "
    "explanations, Markdown, or extra text."
)
ANSWER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
MARKER = "decision"
NOUL_YES = "The answer is yes (affirmative, or align with the claim)."
NOUL_NO = "The answer is no (negative, or disagree with the claim)."
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "qwen3_5_jev.jinja"


def question_options(question: dict) -> list[tuple[str, str]]:
    """按题型展开为有序的 (value, description) 列表。"""
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        return [(str(k), str(v)) for k, v in criteria.items()]
    if kind == "score":
        if isinstance(criteria, list):
            return [(str(i), str(v)) for i, v in enumerate(criteria)]
        return [(str(k), str(v)) for k, v in criteria.items()]
    if kind == "noul":
        descriptions = criteria or {}
        yes = next(
            (str(v) for k, v in descriptions.items() if str(k).lower() in {"yes", "true", "1"}),
            NOUL_YES,
        )
        no = next(
            (str(v) for k, v in descriptions.items() if str(k).lower() in {"no", "false", "0"}),
            NOUL_NO,
        )
        return [("no", no), ("yes", yes)]
    raise ValueError(f"unsupported question type: {kind!r}")


def normalize_label(question: dict, label: object) -> str:
    """noul 的 true/false/1/0 归一到 yes/no，其余题型转字符串。"""
    if question["type"] == "noul":
        return "yes" if str(label).lower() in {"yes", "true", "1"} else "no"
    return str(label)


def compile_record(record: dict, include_targets: bool = True) -> dict:
    """把一条 canonical record 编译成 messages 和符号映射。

    返回 fields / options（field → [(symbol, value, description)]）/ targets（field → 符号）。
    """
    questions = record["questions"]
    fields: list[str] = []
    options: dict[str, list[tuple[str, str, str]]] = {}
    targets: dict[str, str] = {}
    schema_lines: list[str] = []
    for field, question in questions.items():
        opts = question_options(question)
        field = str(field)
        fields.append(field)
        symbols = ANSWER_SYMBOLS[: len(opts)]
        options[field] = [
            (symbol, value, description)
            for symbol, (value, description) in zip(symbols, opts)
        ]
        schema_lines.append(f"{field}: {question['instructions']}")
        for symbol, value, description in options[field]:
            schema_lines.append(f"    {symbol} = {value}: {description}")
        if include_targets:
            label = normalize_label(question, record["targets"][field]["label"])
            values = [value for _, value, _ in options[field]]
            targets[field] = symbols[values.index(label)]

    state = json.dumps(record.get("state"), ensure_ascii=False, indent=2)
    user_text = (
        "Return one answer for every field using the supplied answer symbols.\n\n"
        f"## State\n{state}\n"
        "## Decision schema\n" + "\n".join(schema_lines)
    )
    skeleton = json.dumps(dict.fromkeys(fields, MARKER), ensure_ascii=False, indent=4)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": skeleton},
    ]
    return {
        "messages": messages,
        "fields": fields,
        "options": options,
        "targets": targets if include_targets else None,
    }


class Encoder:
    """绑定 tokenizer 与钉死的 chat template，负责 tokenization 和 marker 定位。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.template = TEMPLATE_PATH.read_text()
        marker_ids = tokenizer.encode(MARKER, add_special_tokens=False)
        assert len(marker_ids) == 1, "marker 必须是单 token"
        self.marker_id = marker_ids[0]

    def encode(self, record: dict, include_targets: bool = True) -> dict:
        """渲染完整训练/推理文本，返回 token ids 与各字段的 marker 索引。

        marker 索引靠"prefix 之外、骨架之内"定位，不扫全文，因此 instructions 或
        state 里出现单词 decision 也不会误判。
        """
        compiled = compile_record(record, include_targets)
        messages = compiled["messages"]
        ids = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(
                messages, chat_template=self.template, tokenize=False
            ),
            add_special_tokens=False,
        )
        prefix_ids = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(
                messages[:-1],
                chat_template=self.template,
                tokenize=False,
                add_generation_prompt=True,
            ),
            add_special_tokens=False,
        )
        assert ids[: len(prefix_ids)] == prefix_ids, "prefix/skeleton 边界 tokenization 不稳定"
        skeleton_ids = ids[len(prefix_ids) :]
        # 骨架里 marker token 的命中包含字段名中的出现（如 ToolACE 的字段就叫 "decision"，
        # 骨架为 {"decision": "decision"}），按字段顺序先跳过 key 里的命中，再取 value。
        hits = [
            i for i, token_id in enumerate(skeleton_ids) if token_id == self.marker_id
        ]
        markers: dict[str, int] = {}
        cursor = 0
        for field in compiled["fields"]:
            cursor += self.tokenizer.encode(field, add_special_tokens=False).count(
                self.marker_id
            )
            markers[field] = len(prefix_ids) + hits[cursor]
            cursor += 1
        assert cursor == len(hits), "骨架中 marker 数与字段数不一致"
        return {
            **compiled,
            "ids": ids,
            "markers": markers,
        }

    def symbol_token_id(self, symbol: str) -> int:
        return self.tokenizer.encode(symbol, add_special_tokens=False)[0]

    def build_sft_datum(self, record: dict) -> trio.Datum:
        """构造 masked SFT 的 Datum：只在 marker 前一位置监督答案符号。"""
        encoded = self.encode(record)
        ids = encoded["ids"]
        target_tokens = list(ids[1:])
        weights = [0.0] * len(target_tokens)
        for field, symbol in encoded["targets"].items():
            position = encoded["markers"][field] - 1
            target_tokens[position] = self.symbol_token_id(symbol)
            weights[position] = 1.0
        return trio.Datum(
            model_input=trio.ModelInput.from_ints(ids[:-1]),
            loss_fn_inputs={
                "target_tokens": np.asarray(target_tokens, dtype=np.int64),
                "weights": np.asarray(weights, dtype=np.float32),
            },
        )


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
