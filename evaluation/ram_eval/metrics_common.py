from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from .generation import expected_image_records
from .io_utils import ensure_dir, prompt_manifest_path, read_json, read_jsonl, seeds_path, write_json, write_jsonl


def torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_prompt_records(manifest_dir: Path, dataset: str, limit_prompts: int | None = None) -> list[dict]:
    path = prompt_manifest_path(manifest_dir, dataset)
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt manifest for {dataset}: {path}")
    rows = list(read_jsonl(path))
    return rows[:limit_prompts] if limit_prompts is not None else rows


def load_seeds(manifest_dir: Path, dataset: str) -> list[int]:
    seeds = [int(seed) for seed in read_json(seeds_path(manifest_dir))["seeds"]]
    return seeds[:1] if dataset == "mscoco" else seeds


def expected_records_for(
    checkpoint: str,
    dataset: str,
    manifest_dir: Path,
    image_root: Path,
    limit_prompts: int | None = None,
) -> list[dict]:
    prompts = load_prompt_records(manifest_dir, dataset, limit_prompts)
    seeds = load_seeds(manifest_dir, dataset)
    return expected_image_records(checkpoint, dataset, prompts, seeds, image_root)


def existing_records(records: list[dict], limit_images: int | None = None) -> list[dict]:
    rows = [row for row in records if Path(row["image_path"]).exists()]
    return rows[:limit_images] if limit_images is not None else rows


def group_by_prompt(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[row["prompt_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["seed"])
    return dict(grouped)


def open_rgb(path: str | Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def metric_output_dir(metric_root: Path, metric_name: str, checkpoint: str, dataset: str) -> Path:
    return ensure_dir(metric_root / metric_name / checkpoint / dataset)


def write_metric_outputs(out_dir: Path, per_image: list[dict], per_prompt: list[dict], summary: dict) -> None:
    write_jsonl(out_dir / "per_image.jsonl", per_image)
    if per_prompt:
        write_jsonl(out_dir / "per_prompt.jsonl", per_prompt)
    write_json(out_dir / "summary.json", summary)


def summarize_scores(scores: list[float]) -> dict:
    import numpy as np

    arr = np.asarray([score for score in scores if np.isfinite(score)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def iter_batches(items: list, batch_size: int) -> Iterator[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def normalize_score_output(value) -> list[float]:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().float().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, tuple):
        return [float(x) for x in value]
    raise TypeError(f"Cannot normalize score output of type {type(value)!r}: {value!r}")


def save_json_line(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
