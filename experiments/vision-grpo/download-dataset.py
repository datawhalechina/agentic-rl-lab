"""下载 GeoQA 数据集到当前案例的 datasets 目录。

运行（仓库根目录）：
    uv run python experiments/vision-grpo/download-dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import load_dataset
from huggingface_hub import snapshot_download

DATASET_ID = "hz2475/geoQA"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "datasets"
TEST_SEED = 42
TEST_SIZE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 GeoQA 数据集")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="geoqa-") as download_dir:
        snapshot_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=download_dir,
            allow_patterns=["data/*.parquet"],
            force_download=args.force,
        )
        parquet_files = sorted((Path(download_dir) / "data").glob("*.parquet"))
        dataset = load_dataset(
            "parquet",
            data_files={"train": [str(path) for path in parquet_files]},
            split="train",
        )

        train_data = dataset.filter(
            lambda split: split == "train",
            input_columns=["original_split"],
            desc="提取 GeoQA train",
        ).remove_columns("original_split")
        test_data = dataset.filter(
            lambda split: split == "test",
            input_columns=["original_split"],
            desc="提取 GeoQA test",
        ).shuffle(seed=TEST_SEED)
        test_data = test_data.select(
            range(len(test_data) - TEST_SIZE, len(test_data))
        ).remove_columns("original_split")

        train_path = output_dir / "train.parquet"
        test_path = output_dir / "test.parquet"
        train_data.to_parquet(train_path)
        test_data.to_parquet(test_path)

    print(f"dataset_dir={output_dir}")
    print(f"train_file={train_path} rows={len(train_data)}")
    print(f"test_file={test_path} rows={len(test_data)}")


if __name__ == "__main__":
    main()
