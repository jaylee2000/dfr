from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from .config import (
    DATASETS,
    DEFAULT_COCO_CAPTIONS_JSON,
    DEFAULT_COCO_IMAGE_ROOT,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_PROMPT_SOURCES,
)
from .io_utils import prompt_manifest_path, read_json, seeds_path, write_json, write_jsonl

DATASET_SEED_OFFSETS = {
    "geneval": 101,
    "drawbench": 202,
    "pickscore": 303,
    "mscoco": 404,
}


def _read_geneval(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            metadata = json.loads(line)
            prompt = metadata["prompt"]
            records.append(
                {
                    "dataset": "geneval",
                    "source_index": source_index,
                    "prompt": prompt,
                    "metadata": metadata,
                }
            )
    return records


def _read_text_prompts(path: Path, dataset: str) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            prompt = line.strip()
            if not prompt:
                continue
            records.append(
                {
                    "dataset": dataset,
                    "source_index": source_index,
                    "prompt": prompt,
                    "metadata": {},
                }
            )
    return records


def _read_mscoco(captions_json: Path, image_root: Path) -> list[dict]:
    data = read_json(captions_json)
    images = {int(item["id"]): item for item in data.get("images", [])}
    first_caption_by_image: dict[int, str] = {}
    for ann in data.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in first_caption_by_image:
            first_caption_by_image[image_id] = ann["caption"].strip()

    records = []
    for source_index, image_id in enumerate(sorted(first_caption_by_image)):
        image_info = images.get(image_id, {})
        file_name = image_info.get("file_name") or f"COCO_val2014_{image_id:012d}.jpg"
        real_image_path = image_root / file_name
        records.append(
            {
                "dataset": "mscoco",
                "source_index": source_index,
                "prompt": first_caption_by_image[image_id],
                "metadata": {"image_id": image_id, "file_name": file_name},
                "real_image_path": str(real_image_path),
            }
        )
    return records


def _sample(records: Sequence[dict], count: int, seed: int, dataset: str) -> list[dict]:
    if count > len(records):
        raise ValueError(f"Requested {count} {dataset} prompts, but only found {len(records)}.")
    rng = random.Random(seed)
    selected = rng.sample(list(records), count)
    output = []
    for prompt_index, record in enumerate(selected):
        row = dict(record)
        row["prompt_index"] = prompt_index
        row["prompt_id"] = f"{dataset}_{prompt_index:05d}"
        output.append(row)
    return output


def prepare_manifests(args: argparse.Namespace) -> None:
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "geneval": args.geneval_count,
        "drawbench": args.drawbench_count,
        "pickscore": args.pickscore_count,
    }
    readers = {
        "geneval": lambda: _read_geneval(DEFAULT_PROMPT_SOURCES["geneval"]),
        "drawbench": lambda: _read_text_prompts(DEFAULT_PROMPT_SOURCES["drawbench"], "drawbench"),
        "pickscore": lambda: _read_text_prompts(DEFAULT_PROMPT_SOURCES["pickscore"], "pickscore"),
    }
    summary = {
        "seed": args.prompt_seed,
        "images_per_prompt": args.images_per_prompt,
        "datasets": {},
    }

    for dataset in (name for name in args.prompts if name != "mscoco"):
        reader = readers[dataset]
        records = reader()
        selected = _sample(records, counts[dataset], args.prompt_seed + DATASET_SEED_OFFSETS[dataset], dataset)
        write_jsonl(prompt_manifest_path(manifest_dir, dataset), selected)
        summary["datasets"][dataset] = {
            "source": str(DEFAULT_PROMPT_SOURCES[dataset]),
            "source_count": len(records),
            "selected_count": len(selected),
        }

    if "mscoco" in args.prompts:
        coco_records = _read_mscoco(DEFAULT_COCO_CAPTIONS_JSON, DEFAULT_COCO_IMAGE_ROOT)
        selected = _sample(coco_records, args.coco_count, args.prompt_seed + DATASET_SEED_OFFSETS["mscoco"], "mscoco")
        write_jsonl(prompt_manifest_path(manifest_dir, "mscoco"), selected)
        summary["datasets"]["mscoco"] = {
            "source": str(DEFAULT_COCO_CAPTIONS_JSON),
            "image_root": str(DEFAULT_COCO_IMAGE_ROOT),
            "source_count": len(coco_records),
            "selected_count": len(selected),
        }

    seeds = [args.seed_base + i for i in range(args.images_per_prompt)]
    write_json(seeds_path(manifest_dir), {"seed_base": args.seed_base, "seeds": seeds})
    write_json(manifest_dir / "manifest_summary.json", summary)

    print(f"Wrote manifests to {manifest_dir}")
    for dataset in args.prompts:
        path = prompt_manifest_path(manifest_dir, dataset)
        print(f"  {dataset}: {'present' if path.exists() else 'missing'} ({path})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create prompt and seed manifests for the RAM evaluation.")
    parser.set_defaults(manifest_dir=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--prompts", nargs="+", choices=DATASETS, required=True)
    parser.add_argument("--geneval-count", type=int, default=256)
    parser.add_argument("--drawbench-count", type=int, default=256)
    parser.add_argument("--pickscore-count", type=int, default=256)
    parser.add_argument("--coco-count", type=int, default=30000)
    parser.add_argument("--prompt-seed", type=int, default=20260529)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--images-per-prompt", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> None:
    prepare_manifests(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
