"""Qwen3.5 的消息模板、图文 chunks、SFT mask 和动作协议。"""

import io
import re
from pathlib import Path

import numpy as np
import pytrio as trio
from PIL import Image

BASE_MODEL = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TEMPLATE = (Path(__file__).parent / "templates/qwen3_5_spec_o3.jinja").read_text()
IMAGE_PAD = "<|image_pad|>"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "spectral_visualization_tool",
            "description": "Visualize a wavelength window from the current spectrum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wavelength_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Wavelength interval [minimum, maximum] in Angstroms",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional spectral feature label",
                    },
                },
                "required": ["wavelength_range"],
            },
        },
    }
]
NUMBER = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def render_messages(messages, tokenizer, generation=False):
    """渲染可阅读文本；generation=True 时追加下一轮 assistant 思考前缀。"""
    return tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        chat_template=TEMPLATE,
        tokenize=False,
        add_generation_prompt=generation,
        enable_thinking=True,
    )


def image_chunk(path, image_processor):
    """把一张图编码为 PNG，并用匹配的处理器计算视觉 token 数。"""
    # 原图中透明背景合成到白色；传入 processor 和服务端的是同一张 RGB 图。
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        image = Image.alpha_composite(
            Image.new("RGBA", rgba.size, "white"),
            rgba,
        ).convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    patches = image_processor.get_number_of_image_patches(
        image.height,
        image.width,
        images_kwargs={},
    )
    return trio.ImageChunk(
        data=buffer.getvalue(),
        format="png",
        expected_tokens=patches // image_processor.merge_size ** 2,
    )


def pack_chunks(ids, masks, image_paths, tokenizer, image_processor):
    """image_pad 展开成视觉 token；target 和 mask 随之扩展。"""
    image_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    paths = iter(image_paths)
    chunks, targets, weights = [], [], []
    start = 0

    for index, token in enumerate(ids):
        if token == image_id:
            # 先保留图片前面的文本，以及模板给出的 assistant 监督范围。
            chunks.append(trio.types.EncodedTextChunk(tokens=ids[start:index]))
            targets.extend(ids[start:index])
            weights.extend(masks[start:index])

            # 一枚 image_pad 替换为实际图片；所有视觉位置只提供上下文。
            chunk = image_chunk(next(paths), image_processor)
            chunks.append(chunk)
            targets.extend([0] * chunk.expected_tokens)
            weights.extend([0.0] * chunk.expected_tokens)
            start = index + 1

    # 最后一张图片后的文本也要进入输入和监督序列。
    chunks.append(trio.types.EncodedTextChunk(tokens=ids[start:]))
    targets.extend(ids[start:])
    weights.extend(masks[start:])
    return (
        trio.ModelInput(chunks=chunks),
        np.asarray(targets, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def encode_messages(messages, image_paths, tokenizer, image_processor, generation=False):
    """从模板取得 token 与 assistant mask，再把图片占位符展开为图文 chunks。"""
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        chat_template=TEMPLATE,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=generation,
        enable_thinking=True,
    )
    return pack_chunks(
        encoded["input_ids"],
        encoded["assistant_masks"],
        image_paths,
        tokenizer,
        image_processor,
    )


def build_sft_datum(row, tokenizer, image_processor):
    """构造完整多轮轨迹的 SFT 输入，仅监督 assistant 生成的位置。"""
    model_input, targets, weights = encode_messages(
        row["messages"],
        row["images"],
        tokenizer,
        image_processor,
    )

    # 最后一个文本 token 只作为 target，所有序列统一右移一次。
    chunks = model_input.chunks[:-1] + [
        trio.types.EncodedTextChunk(tokens=model_input.chunks[-1].tokens[:-1]),
    ]
    targets[weights == 0] = 0
    return trio.Datum(
        model_input=trio.ModelInput(chunks=chunks),
        loss_fn_inputs={
            "target_tokens": targets[1:],
            "weights": weights[1:],
        },
    )


def build_rl_datum(trajectory, advantage):
    """将实际采样轨迹右移一位，只在 assistant 动作位置计算 PPO loss。"""
    chunks = trajectory["model_input"].chunks
    model_input = trio.ModelInput(chunks=[
        *chunks[:-1],
        trio.types.EncodedTextChunk(tokens=chunks[-1].tokens[:-1]),
    ])
    action_mask = np.asarray(trajectory["action_mask"], dtype=np.float32)[1:]

    return trio.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": np.asarray(trajectory["target_tokens"], dtype=np.int64)[1:],
            "logprobs": np.asarray(trajectory["logprobs"], dtype=np.float32)[1:],
            "advantages": action_mask * advantage,
        },
    )


def tool_continuation(messages, image_paths, tokenizer, image_processor):
    """只编码模板中新追加的 tool 返回和 assistant 前缀，保留旧采样 token。"""
    rendered = render_messages(messages, tokenizer, generation=True)
    suffix = rendered[rendered.rindex("<|im_start|>user\n<tool_response>") :]
    ids = tokenizer.encode(suffix, add_special_tokens=False)
    return pack_chunks(
        ids,
        [0] * len(ids),
        image_paths,
        tokenizer,
        image_processor,
    )[0]


def parse_action(text):
    """动作语法属于环境规则；非法动作结束该轨迹，不重试或补写。"""
    body = (
        text.partition("</think>")[2]
        .strip()
        .removesuffix("<|im_end|>")
        .strip()
    )

    # 最终回答必须使用约定的 answer / boxed 格式。
    answer = re.fullmatch(r"<answer>\s*\\boxed\{(YES|NO)\}(.*?)</answer>", body, re.S)
    if answer:
        return {"kind": "answer", "prediction": answer[1]}

    # 工具动作使用 Qwen3.5 原生 XML，从参数中解析波长区间和可选标签。
    call = re.fullmatch(
        r"<tool_call>\s*<function=spectral_visualization_tool>"
        r"\s*(.*?)\s*</function>\s*</tool_call>",
        body,
        re.S,
    )
    if call:
        parameters = dict(
            re.findall(
                r"<parameter=(\w+)>\s*(.*?)\s*</parameter>",
                call[1],
                re.S,
            )
        )
        interval = re.fullmatch(
            rf"\[\s*({NUMBER})\s*,\s*({NUMBER})\s*\]",
            parameters.get("wavelength_range", ""),
        )
        if interval:
            return {
                "kind": "tool",
                "wavelength_range": [float(interval[1]), float(interval[2])],
                "label": parameters.get("label", ""),
            }

    return {"kind": "invalid"}
