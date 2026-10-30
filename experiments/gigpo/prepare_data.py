"""Download official ALFWorld data and fingerprint the exact filtered splits."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from data import default_data_root, discover_games
from experiment import BASE_DIR, dataset_fingerprint, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--manifest", type=Path, default=BASE_DIR / "outputs" / "data-manifest.json")
    parser.add_argument("--allow-different-counts", action="store_true")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if args.download:
        executable = shutil.which("alfworld-download")
        if executable is None:
            raise SystemExit("Install the optional dependencies: uv sync --extra alfworld")
        subprocess.run([executable, "--data-dir", str(root)], check=True)
    expected = {"train": 3553, "valid_seen": 140, "valid_unseen": 134}
    manifest = {"data_root": str(root), "splits": {}}
    for split, count in expected.items():
        games = discover_games(root, split)
        if len(games) != count and not args.allow_different_counts:
            raise ValueError(f"{split}: expected {count} games, got {len(games)}; check dataset version or use --allow-different-counts explicitly")
        manifest["splits"][split] = {
            "count": len(games), "fingerprint": dataset_fingerprint(games),
            "task_counts": dict(Counter(game.task_type for game in games)),
            "game_ids": [game.id for game in games],
        }
        print(f"{split}: {len(games)} games")
    write_json(args.manifest, manifest)
    print(f"Saved data manifest: {args.manifest}")


if __name__ == "__main__":
    main()
