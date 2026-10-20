"""Average per-prompt empirical FD to the base checkpoint (historical CLI: fid)."""

from __future__ import annotations

import gc
from pathlib import Path

from .fd_encoders import FD_ENCODERS, FdEncoder
from .model_assets import ModelLock, enforce_offline_mode
from .metrics_common import (
    existing_records,
    expected_records_for,
    group_by_prompt,
    metric_output_dir,
    summarize_scores,
    write_json,
    write_jsonl,
)


def frechet_from_features(a, b) -> float:
    """Exact empirical FD using a sample-space SVD, including singular covariances."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or len(a) < 2 or len(b) < 2:
        raise ValueError(f"FD needs two feature arrays with >=2 rows and equal dimensions; got {a.shape}, {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("FD features must be finite")
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    cross = (ac @ bc.T) / np.sqrt((len(a) - 1) * (len(b) - 1))
    covariance_term = np.square(ac).sum() / (len(a) - 1) + np.square(bc).sum() / (len(b) - 1)
    value = np.square(a.mean(axis=0) - b.mean(axis=0)).sum() + covariance_term
    value -= 2.0 * np.linalg.svd(cross, compute_uv=False).sum()
    return float(max(value, 0.0))


def run(args) -> None:
    metric_root = Path(args.metric_root)
    manifest_dir = Path(args.manifest_dir)
    image_root = Path(args.image_root)
    workloads = {}
    for dataset in args.datasets:
        for checkpoint in dict.fromkeys([args.base_checkpoint, *args.checkpoints]):
            records = existing_records(
                expected_records_for(checkpoint, dataset, manifest_dir, image_root, args.limit_prompts),
                args.limit_images,
                allow_missing=args.dry_run,
            )
            workloads[checkpoint, dataset] = group_by_prompt(records)
            if args.dry_run:
                print(f"[dry-run] FD {checkpoint}/{dataset}: {len(records)} existing images")
    if args.dry_run:
        return

    enforce_offline_mode()
    lock = ModelLock.load(args.model_lock)
    summaries = {(checkpoint, dataset): {} for checkpoint in args.checkpoints for dataset in args.datasets}
    for encoder_name, spec in FD_ENCODERS.items():
        encoder = FdEncoder(encoder_name, lock.metric_path(encoder_name), args.device)
        for dataset in args.datasets:
            base_groups = workloads[args.base_checkpoint, dataset]
            # Retain only one dataset/encoder's base features, and reuse them for every checkpoint.
            base_features = {
                prompt_id: encoder.extract(rows, args.batch_size) for prompt_id, rows in base_groups.items()
            }
            for checkpoint in args.checkpoints:
                per_prompt = []
                for prompt_id, rows in workloads[checkpoint, dataset].items():
                    reference = base_groups[prompt_id]
                    if [row["seed"] for row in rows] != [row["seed"] for row in reference]:
                        raise ValueError(f"Mismatched base image seeds: {checkpoint}/{dataset}/{prompt_id}")
                    features = (
                        base_features[prompt_id]
                        if checkpoint == args.base_checkpoint
                        else encoder.extract(rows, args.batch_size)
                    )
                    fd = frechet_from_features(features, base_features[prompt_id])
                    per_prompt.append(
                        {
                            "checkpoint": checkpoint,
                            "dataset": dataset,
                            "encoder": encoder_name,
                            "prompt_id": prompt_id,
                            "prompt_index": rows[0]["prompt_index"],
                            "prompt": rows[0]["prompt"],
                            "fd_to_base": fd,
                            "count": len(rows),
                            "base_count": len(reference),
                        }
                    )
                summary = summarize_scores([row["fd_to_base"] for row in per_prompt])
                summaries[checkpoint, dataset][encoder_name] = summary
                out = metric_output_dir(metric_root, "fid", checkpoint, dataset) / encoder_name
                write_jsonl(out / "per_prompt.jsonl", per_prompt)
                write_json(
                    out / "summary.json",
                    {
                        "checkpoint": checkpoint,
                        "dataset": dataset,
                        "encoder": encoder_name,
                        "feature_source": spec.source,
                        "base_checkpoint": args.base_checkpoint,
                        "fd_to_base": summary,
                    },
                )
                print(f"Wrote {out}")
            del base_features
        del encoder
        gc.collect()
        if args.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    for (checkpoint, dataset), values in summaries.items():
        out = metric_output_dir(metric_root, "fid", checkpoint, dataset)
        write_json(
            out / "summary.json",
            {"checkpoint": checkpoint, "dataset": dataset, "base_checkpoint": args.base_checkpoint, "encoders": values},
        )
