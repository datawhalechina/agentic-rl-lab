"""通过 Hugging Face 官方 HTTP 下载并整理 cold-start 轨迹与 SpecVI-Bench。

在 experiments/spec-o3/ 目录下依次运行：
    uv run python prepare_data.py sft    # 下载并整理到 datasets/sft/
    uv run python prepare_data.py bench  # 下载并整理到 datasets/rl/

bench 会读取整理好的 SFT 数据，划分开发集时排除 SFT 中出现过的光谱对象。
SFT 原始对话和整理结果保存在 datasets/sft/，图片下载到 datasets/sft/images/。
RL 原始 Parquet 下载到 datasets/rl/data/，图片提取到 datasets/rl/images/，
光谱数组保存到 datasets/rl/spectra/，划分结果保存在 datasets/rl/。
下载通过 local_dir 直接写入对应目录，重新运行会复用已完成的文件。
已有原始 SFT 目录可用 --sft-source 导入；文件会复制到 datasets/sft/。
关闭 Xet，默认同时下载 8 个文件；可用 --download-workers 调整并发数。
旧下载进程先 Ctrl+C 停止，再运行以上命令，让新设置生效。
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import shutil

# 在导入 Hugging Face 库之前设置，同时用于数据、tokenizer 和 image processor。
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from protocol import BASE_MODEL, MODEL_REVISION, build_sft_datum

SFT_REPO = "Maxwell-Jia/Spec-o3-ColdStartSFT"
SFT_REVISION = "7ea73e408ffe6a0b25e5f738e517a71217eb7fbc"
BENCH_REPO = "Maxwell-Jia/SpecVI-Bench"
BENCH_REVISION = "0e003e00ac128c665acd702af3dc0992c4161637"
TASK_NAMES = {
    "Cataclysmic": "cv",
    "Carbon Star": "cc",
    "S-type Star": "ss",
    "M-type Giant": "gm",
    "White Dwarf": "wd",
}
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
THINK = re.compile(r"<think>(.*?)</think>", re.S)
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def repair_text(text):
    """只修复已观察到、能确定原意的编码；不猜测丢失的希腊字母。"""
    # 这些替换有先后顺序：先处理具体的光谱符号，再标记剩余控制字符。
    text = text.replace("\x05saw\x0btooth\r", '"saw-tooth"')
    text = re.sub(r"\x1b\[0;31m", "", text)
    text = re.sub(
        r"C(?:\x00(?:02|82|b2)|\x02\x082|\x08(?:2|\x02)?|\x02|\x178|\x192)",
        "C₂",
        text,
    )
    text = text.replace("\x0394", "Δ").replace("\x03bb", "λ")
    text = re.sub(
        r"(?:\x03b?|\x00b)([123])",
        lambda m: {"1": "α", "2": "β", "3": "γ"}[m[1]],
        text,
    )

    # 氢谱线、区间符号与单位。
    text = re.sub(r"H\x01b?", "Hα", text)
    text = re.sub(r"H\x02b?", "Hβ", text)
    text = text.replace("H\x1b", "Hβ")
    text = re.sub(r"H\x03(?![b0-9])", "Hγ", text)
    text = re.sub(
        r"(?<=\d)(?:\x05\x17|\x0213|[\x01\x13\x14\x15])(?=\d)",
        "–",
        text,
    )
    text = text.replace("\x02273", "≳").replace("\x02248", "≈")
    text = re.sub(r"\x01b?(?!\d)", "α", text)
    text = re.sub(r"\x02b?(?!\d)", "β", text)
    text = re.sub(r"\x00(?:c5|5)|\x0c5|\x1c5|\x178|\x192|\x05|\x15", "Å", text)
    text = re.sub(r"(?<=[\d$] )[\x16\x1e]", "Å", text)
    text = re.sub(r"ngstr(?:\x00f6?|[\x0f\x1f]6?)ms", "ngströms", text)

    # 已确认的标点编码。
    text = text.replace("\x0cchopped up\x0c", '"chopped up"')
    text = text.translate(str.maketrans({
        "\x11": "-",
        "\x13": "–",
        "\x14": "—",
        "\x19": "’",
        "\x1c": "“",
        "\x1d": "”",
    }))

    # 如 \x03b 已丢失末位，无法分辨 α/β/γ。标明原码位，另存逐条清理记录。
    return CONTROL.sub(lambda m: f"[U+{ord(m[0]):04X}]", text)


def multimodal_content(text, image_paths):
    """按 image 占位符出现的顺序，交错放入文本和对应图片路径。"""
    content = []
    for index, part in enumerate(text.split("<image>")):
        if index:
            content.append({"type": "image", "image": next(image_paths)})
        if part:
            content.append({"type": "text", "text": repair_text(part)})
    return content


def convert_sft(sample, image_root):
    """将一条 ShareGPT 轨迹转成包含 reasoning、工具调用和图片的 messages。"""
    paths = [str(image_root / path) for path in sample["images"]]
    image_paths = iter(paths)
    messages = []

    for message in sample["conversations"]:
        role, text = message["from"], message["value"]

        if role == "system":
            # 工具说明由 Qwen3.5 模板生成，保留原 system 中的任务说明。
            messages.append({
                "role": "system",
                "content": text.split("\n# Tools", 1)[0].strip(),
            })

        elif role == "human":
            # 原数据把用户问题和工具返回都放在 human 中，需要区分角色。
            is_tool = text.strip().startswith("<tool_response>")
            if is_tool:
                text = (
                    text.strip()
                    .removeprefix("<tool_response>")
                    .removesuffix("</tool_response>")
                    .strip()
                )
            messages.append({
                "role": "tool" if is_tool else "user",
                "content": multimodal_content(text, image_paths),
            })

        elif role == "gpt":
            # 分开保存思考、结构化工具调用和最终回答，由模板统一渲染。
            thoughts = THINK.findall(text)
            calls = []
            for raw in TOOL_CALL.findall(text):
                # 公开 JSON 的工具参数字符串也含控制字符，先解析，再修复 label。
                call = json.loads(raw, strict=False)
                if "label" in call["arguments"]:
                    call["arguments"]["label"] = repair_text(call["arguments"]["label"])
                calls.append({"type": "function", "function": call})

            messages.append({
                "role": "assistant",
                "reasoning_content": repair_text("".join(thoughts).strip()),
                "content": repair_text(TOOL_CALL.sub("", THINK.sub("", text)).strip()),
                "tool_calls": calls,
            })

    question = sample["conversations"][1]["value"].splitlines()[1]
    task = next(code for name, code in TASK_NAMES.items() if name in question)
    return {
        "id": f"{task}-{sample['source_name']}-{sample['sample_index']}",
        "task": task,
        "source_name": sample["source_name"],
        "label": sample["ground_truth"],
        "messages": messages,
        "images": paths,
    }


def split_sources(rows, fraction, seed):
    """按光谱对象划分，保证同一对象的多条轨迹进入同一个子集。"""
    sources = sorted({row["source_name"] for row in rows})
    random.Random(seed).shuffle(sources)
    return set(sources[: round(len(sources) * fraction)])


def write_jsonl(path, rows):
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_sft(args):
    """下载原始轨迹，构造图文训练样本，输出划分结果和清理统计。"""
    # 1. 下载到 datasets/sft/，或从已有原始目录导入。
    root = args.output / "sft"
    root.mkdir(parents=True, exist_ok=True)
    if args.sft_source:
        shutil.copytree(args.sft_source, root, dirs_exist_ok=True)
    else:
        snapshot_download(
            SFT_REPO,
            repo_type="dataset",
            revision=SFT_REVISION,
            local_dir=root,
            max_workers=args.download_workers,
        )

    # 2. 转换完整轨迹，并用实际图文 Datum 统计长度和监督 token 数。
    raw_rows = json.loads((root / "train_sharegpt.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        revision=MODEL_REVISION,
    )
    processor = AutoImageProcessor.from_pretrained(
        BASE_MODEL,
        revision=MODEL_REVISION,
        backend="pil",
    )
    rows, cleaning = [], []

    for index, sample in enumerate(tqdm(raw_rows, desc="SFT messages + image tokens")):
        row = convert_sft(sample, root.resolve())
        datum = build_sft_datum(row, tokenizer, processor)
        image_tokens = [
            chunk.expected_tokens
            for chunk in datum.model_input.chunks
            if chunk.type == "image"
        ]
        row["length"] = len(datum.model_input)
        row["image_tokens"] = image_tokens
        row["supervised_tokens"] = int(datum.loss_fn_inputs["weights"].to_numpy().sum())
        rows.append(row)

        # 单独记录原始控制字符和修复结果，便于人工审阅。
        for turn, (old, new) in enumerate(zip(sample["conversations"], row["messages"])):
            if CONTROL.search(old["value"]):
                cleaning.append({
                    "row": index,
                    "turn": turn,
                    "before": old["value"],
                    "after": new,
                })

    # 3. 按光谱对象划分 train / val，保存转换后的数据和清理记录。
    validation_sources = split_sources(rows, args.val_fraction, args.seed)
    train = [row for row in rows if row["source_name"] not in validation_sources]
    val = [row for row in rows if row["source_name"] in validation_sources]
    write_jsonl(root / "sft_train.jsonl", train)
    write_jsonl(root / "sft_val.jsonl", val)
    write_jsonl(root / "sft_cleaning.jsonl", cleaning)

    # 4. 汇总任务分布、图文长度和仍待人工判断的符号。
    lengths = np.asarray([row["length"] for row in rows])
    unresolved = [
        row["id"]
        for row in rows
        if "[U+" in json.dumps(row["messages"])
    ]
    stats = {
        "dataset_revision": SFT_REVISION,
        "model_revision": MODEL_REVISION,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train": len(train),
        "val": len(val),
        "tasks_train": dict(Counter(row["task"] for row in train)),
        "tasks_val": dict(Counter(row["task"] for row in val)),
        "assistant_turns": sum(
            m["role"] == "assistant"
            for row in rows
            for m in row["messages"]
        ),
        "images": sum(len(row["images"]) for row in rows),
        "text_tokens": sum(
            row["length"] + 1 - sum(row["image_tokens"])
            for row in rows
        ),
        "visual_tokens": sum(sum(row["image_tokens"]) for row in rows),
        "length_p50_p90_p99_max": np.percentile(lengths, [50, 90, 99, 100]).tolist(),
        "over_16384": int((lengths > 16384).sum()),
        "changed_messages": len(cleaning),
        "unresolved_symbol_rows": unresolved,
    }
    (root / "sft_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(
        {k: v for k, v in stats.items() if k != "unresolved_symbol_rows"},
        ensure_ascii=False,
        indent=2,
    ))
    print(f"SFT 数据目录：{root.resolve()}")
    print(f"仍含无法确定符号的轨迹：{len(unresolved)}；见该目录的 sft_cleaning.jsonl / sft_stats.json")


def prepare_bench(args):
    """整理 benchmark 的图片和光谱数组，划分 RL 训练集、开发集和测试集。"""
    # 1. 下载原始 Parquet，在 datasets/rl/ 下分别保存图片和光谱数组。
    root = Path(snapshot_download(
        BENCH_REPO,
        repo_type="dataset",
        revision=BENCH_REVISION,
        allow_patterns="data/*.parquet",
        local_dir=args.output / "rl",
        max_workers=args.download_workers,
    ))
    images_dir = root / "images"
    spectra_dir = root / "spectra"
    images_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)
    partitions = {"train": [], "test": []}

    # 2. 按 split → 任务文件 → batch → 样本的顺序转换，逐批读取 Parquet。
    for split in partitions:
        for path in sorted((root / "data").glob(f"{split}_*.parquet")):
            task = path.stem.split("_", 1)[1]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                for sample in batch.to_pylist():
                    info = sample["extra_info"]
                    name = f"{split}-{task}-{info['index']}"

                    # 提取初始图片，并按占位符顺序构造输入消息。
                    images = []
                    for index, image in enumerate(sample["images"]):
                        image_path = images_dir / f"{name}-{index}.png"
                        image_path.write_bytes(image["bytes"])
                        images.append(str(image_path.resolve()))
                    image_paths = iter(images)
                    messages = [
                        {
                            "role": m["role"],
                            "content": multimodal_content(m["content"], image_paths),
                        }
                        for m in sample["prompt"]
                    ]

                    # 保存波长、通量和红移，供 rollout 中的绘图工具使用。
                    spectrum_path = spectra_dir / f"{name}.npz"
                    spectrum = info["tools_kwargs"]["spectral_visualization_tool"][
                        "create_kwargs"
                    ]
                    # 公开数据的 redshift=null 表示不作红移校正，沿用作者工具语义。
                    spectrum["redshift"] = (
                        0.0 if spectrum["redshift"] is None else spectrum["redshift"]
                    )
                    np.savez_compressed(spectrum_path, **spectrum)
                    partitions[split].append({
                        "id": name,
                        "task": task,
                        "source_name": info["name"],
                        "label": sample["reward_model"]["ground_truth"],
                        "messages": messages,
                        "images": images,
                        "spectrum": str(spectrum_path.resolve()),
                    })
            print(f"已整理 {path.name}")

    # 3. 开发集只从 SFT 未出现的光谱对象中抽取，其余 train 用于 RL。
    sft_sources = {
        json.loads(line)["source_name"]
        for split in ("train", "val")
        for line in (args.output / "sft" / f"sft_{split}.jsonl").read_text().splitlines()
    }
    unseen = [
        row
        for row in partitions["train"]
        if row["source_name"] not in sft_sources
    ]
    dev_sources = split_sources(unseen, args.val_fraction, args.seed)
    dev = [row for row in partitions["train"] if row["source_name"] in dev_sources]
    train = [row for row in partitions["train"] if row["source_name"] not in dev_sources]

    for name, rows in {"rl_train": train, "dev": dev, "test": partitions["test"]}.items():
        write_jsonl(root / f"bench_{name}.jsonl", rows)

    # 4. 汇总划分数量、任务分布以及 SFT 与开发集 / 测试集的对象重叠。
    stats = {
        "dataset_revision": BENCH_REVISION,
        "seed": args.seed,
        "rl_train": len(train),
        "dev": len(dev),
        "test": len(partitions["test"]),
        "sft_dev_source_overlap": len(sft_sources & dev_sources),
        "sft_test_source_overlap": len(
            sft_sources & {row["source_name"] for row in partitions["test"]}
        ),
        "tasks_dev": dict(Counter(row["task"] for row in dev)),
    }
    (root / "bench_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"RL 数据目录：{root.resolve()}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # 选择数据准备阶段与原始文件的保存位置。
    parser.add_argument("stage", choices=["sft", "bench"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "datasets",
        help="数据根目录，下面分别创建 sft/ 和 rl/",
    )
    parser.add_argument(
        "--sft-source",
        type=Path,
        help="将已有的原始 SFT 数据目录复制到输出目录的 sft/ 下",
    )

    # 下载并发与按对象划分数据的比例 / 随机种子。
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="同时下载的文件数",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"下载地址：{os.environ['HF_ENDPOINT']}；"
        f"通道：HTTP（Xet 已关闭）；下载并发数：{args.download_workers}",
        flush=True,
    )
    if args.stage == "sft":
        prepare_sft(args)
    else:
        prepare_bench(args)
