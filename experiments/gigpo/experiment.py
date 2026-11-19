"""Local reproducibility manifests and checkpoint discovery (no network calls)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
UPSTREAM_COMMIT = "bbab8c98761b5e9e5d57f4b4bb899c16dcec0034"
OFFICIAL_GIGPO_COMMIT = "20bd331bdbc9026a5668e11362178e10ab7400c8"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_fingerprint(games: list[Any]) -> str:
    """Include ordered IDs and file bytes; paths can differ across machines."""
    digest = hashlib.sha256()
    for game in games:
        for value in (game.id, sha256(game.game_file), sha256(game.game_file.with_name("traj_data.json"))):
            digest.update(value.encode("utf-8") + b"\0")
    return digest.hexdigest()


def provenance() -> dict[str, Any]:
    versions = {}
    for package in ("pytrio", "alfworld", "textworld", "swanlab", "numpy", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "upstream_commit": UPSTREAM_COMMIT, "official_gigpo_commit": OFFICIAL_GIGPO_COMMIT,
        "python": platform.python_version(), "versions": versions,
        "source_sha256": {path.name: sha256(path) for path in sorted(BASE_DIR.glob("*.py"))},
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sampler_from_manifest(path: Path, step: int) -> str:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("type") != "training_run":
        raise ValueError("Expected a training run manifest")
    matches = [item for item in manifest["checkpoints"] if item["update"] == step]
    if not matches:
        raise ValueError(f"No checkpoint for update {step}; available: {[x['update'] for x in manifest['checkpoints']]}")
    result = matches[-1]["sampler_weights"]
    if not isinstance(result, str) or not result.startswith("trio://"):
        raise ValueError("Invalid sampler weights path in manifest")
    return result
