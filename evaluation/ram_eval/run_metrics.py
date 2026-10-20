from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

from .config import (
    DATASETS,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_METRIC_ROOT,
    DEFAULT_MODEL_LOCK,
)
from .metrics_common import existing_records, expected_records_for


METRICS = ("pickscore", "imagereward", "fid", "dreamsim")
METRIC_MODULES = {
    name: f"ram_eval.metric_{name}" for name in METRICS
}


def _generation_configs(image_root: Path) -> list[tuple[Path, dict]]:
    if not image_root.is_dir():
        raise ValueError(f"Image path root does not exist: {image_root}")
    configs = []
    for child in sorted(image_root.iterdir()):
        config_path = child / "generation_config.json"
        if not child.is_dir() or not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configs.append((config_path, config))
    if not configs:
        raise ValueError(f"No generated checkpoints found under {image_root}")
    return configs


def discover_checkpoints(image_root: Path) -> tuple[list[str], str | None]:
    checkpoints = []
    detected_base = None
    for config_path, config in _generation_configs(image_root):
        child = config_path.parent
        checkpoint = config.get("checkpoint", {})
        key = checkpoint.get("key")
        if key != child.name:
            raise ValueError(f"Invalid checkpoint metadata in {config_path}")
        checkpoints.append(key)
        if checkpoint.get("loader") == "base":
            if detected_base is not None:
                raise ValueError(f"Multiple base checkpoints found under {image_root}")
            detected_base = key
    return checkpoints, detected_base


def discover_prompts(image_root: Path, checkpoints: list[str] | None = None) -> list[str]:
    discovered = None
    for config_path, config in _generation_configs(image_root):
        if checkpoints is not None and config_path.parent.name not in checkpoints:
            continue
        prompts = config.get("prompts")
        if not isinstance(prompts, list) or not prompts or any(
            prompt not in DATASETS for prompt in prompts
        ):
            raise ValueError(f"Invalid prompt metadata in {config_path}")
        if discovered is None:
            discovered = prompts
        elif prompts != discovered:
            raise ValueError(
                "Generated checkpoints use different prompt sets; pass --prompts explicitly"
            )
    assert discovered is not None
    return discovered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evaluation metrics over generated images.")
    parser.set_defaults(
        image_path_root=str(DEFAULT_IMAGE_ROOT),
        manifest_dir=str(DEFAULT_MANIFEST_DIR),
        metric_root=str(DEFAULT_METRIC_ROOT),
        model_lock=str(DEFAULT_MODEL_LOCK),
    )
    parser.add_argument("--image-path-root")
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--prompts", nargs="+", choices=DATASETS)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    image_root = Path(args.image_path_root)
    discovered, detected_base = discover_checkpoints(image_root)
    checkpoints = args.checkpoints or discovered
    unknown = set(checkpoints).difference(discovered)
    if unknown:
        raise ValueError(f"Generated checkpoints not found: {', '.join(sorted(unknown))}")
    prompts = args.prompts or discover_prompts(image_root, checkpoints)
    base_checkpoint = args.base_checkpoint or detected_base
    if "fid" in args.metrics:
        if base_checkpoint is None:
            raise ValueError(
                "FID for per-prompt datasets requires a generated base checkpoint; "
                "generate one or pass --base-checkpoint"
            )
        if base_checkpoint not in discovered:
            raise ValueError(f"Base checkpoint images not found: {base_checkpoint}")

    # Validate all selected groups before loading any metric models or writing results.
    required_checkpoints = list(checkpoints)
    if "fid" in args.metrics and base_checkpoint not in required_checkpoints:
        required_checkpoints.append(base_checkpoint)
    for checkpoint in required_checkpoints:
        for dataset in prompts:
            existing_records(
                expected_records_for(
                    checkpoint, dataset, Path(args.manifest_dir), image_root, args.limit_prompts
                ),
                args.limit_images,
                allow_missing=args.dry_run,
            )

    metric_args = argparse.Namespace(
        manifest_dir=args.manifest_dir,
        image_root=str(image_root),
        metric_root=args.metric_root,
        checkpoints=checkpoints,
        datasets=prompts,
        base_checkpoint=base_checkpoint,
        model_lock=args.model_lock,
        limit_prompts=args.limit_prompts,
        limit_images=args.limit_images,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        dry_run=args.dry_run,
    )
    for metric in args.metrics:
        print(f"Running metric: {metric}")
        import_module(METRIC_MODULES[metric]).run(metric_args)


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
