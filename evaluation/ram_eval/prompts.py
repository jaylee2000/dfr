from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from .config import (
    DATASETS,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_PROMPT_SOURCES,
)
from .io_utils import prompt_manifest_path, seeds_path, write_json, write_jsonl

DATASET_SEED_OFFSETS = {
    "geneval": 101,
    "drawbench": 202,
    "ocr": 505,
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


def _sample(records: Sequence[dict], count: int, seed: int, dataset: str) -> list[dict]:
    if count < 1:
        raise ValueError(f"Requested {count} {dataset} prompts; count must be positive.")
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
    if args.images_per_prompt < 2:
        raise ValueError("images_per_prompt must be at least 2 for FD and diversity.")
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "geneval": args.geneval_count,
        "drawbench": args.drawbench_count,
        "ocr": args.ocr_count,
    }
    readers = {
        "geneval": lambda: _read_geneval(DEFAULT_PROMPT_SOURCES["geneval"]),
        "drawbench": lambda: _read_text_prompts(DEFAULT_PROMPT_SOURCES["drawbench"], "drawbench"),
        "ocr": lambda: _read_text_prompts(DEFAULT_PROMPT_SOURCES["ocr"], "ocr"),
    }
    summary = {
        "seed": args.prompt_seed,
        "images_per_prompt": args.images_per_prompt,
        "datasets": {},
    }

    for dataset in args.prompts:
        reader = readers[dataset]
        records = reader()
        selected = _sample(records, counts[dataset], args.prompt_seed + DATASET_SEED_OFFSETS[dataset], dataset)
        write_jsonl(prompt_manifest_path(manifest_dir, dataset), selected)
        summary["datasets"][dataset] = {
            "source": str(DEFAULT_PROMPT_SOURCES[dataset]),
            "source_count": len(records),
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
    parser = argparse.ArgumentParser(description="Create prompt and seed manifests for the DFR evaluation.")
    parser.set_defaults(manifest_dir=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--prompts", nargs="+", choices=DATASETS, required=True)
    parser.add_argument("--geneval-count", type=int, default=256)
    parser.add_argument("--drawbench-count", type=int, default=256)
    parser.add_argument("--ocr-count", type=int, default=256)
    parser.add_argument("--prompt-seed", type=int, default=20260529)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--images-per-prompt", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> None:
    prepare_manifests(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
