from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, Iterator


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def slugify(text: str, max_length: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "item")[:max_length]


def image_path(image_root: Path, checkpoint_key: str, dataset: str, prompt_id: str, seed: int) -> Path:
    return image_root / checkpoint_key / dataset / prompt_id / f"seed_{seed:06d}.png"


def image_manifest_path(manifest_dir: Path, checkpoint_key: str, dataset: str) -> Path:
    return manifest_dir / "images" / f"{checkpoint_key}_{dataset}.jsonl"


def prompt_manifest_path(manifest_dir: Path, dataset: str) -> Path:
    return manifest_dir / "prompts" / f"{dataset}.jsonl"


def seeds_path(manifest_dir: Path) -> Path:
    return manifest_dir / "seeds.json"

