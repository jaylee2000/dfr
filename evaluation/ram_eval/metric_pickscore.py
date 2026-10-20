from __future__ import annotations

from pathlib import Path

from .config import PER_PROMPT_DATASETS
from .model_assets import ModelLock, enforce_offline_mode
from .metrics_common import (
    existing_records,
    expected_records_for,
    iter_batches,
    metric_output_dir,
    open_rgb,
    save_json_line,
    summarize_scores,
    torch_dtype,
    write_json,
    write_jsonl,
)


def _score_batch(model, processor, records: list[dict], device: str, dtype_name: str) -> list[float]:
    import torch

    images = [open_rgb(row["image_path"]) for row in records]
    prompts = [row["prompt"] for row in records]
    image_inputs = processor(images=images, padding=True, truncation=True, max_length=77, return_tensors="pt")
    text_inputs = processor(text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
    dtype = torch_dtype(dtype_name)
    image_inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
        for k, v in image_inputs.items()
    }
    text_inputs = {k: v.to(device=device) for k, v in text_inputs.items()}
    with torch.inference_mode():
        image_embs = model.get_image_features(**image_inputs).to(dtype=torch.float32)
        text_embs = model.get_text_features(**text_inputs).to(dtype=torch.float32)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        scores = model.logit_scale.exp().float() * (text_embs @ image_embs.T).diag()
        scores = scores / 26.0
    del images, image_inputs, text_inputs
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return [float(x) for x in scores.detach().cpu().tolist()]


def run(args) -> None:
    metric_root = Path(args.metric_root)
    manifest_dir = Path(args.manifest_dir)
    image_root = Path(args.image_root)
    if args.dry_run:
        for checkpoint in args.checkpoints:
            for dataset in args.datasets:
                records = existing_records(
                    expected_records_for(checkpoint, dataset, manifest_dir, image_root, args.limit_prompts),
                    args.limit_images,
                    allow_missing=True,
                )
                print(f"[dry-run] PickScore {checkpoint}/{dataset}: {len(records)} existing images")
        return

    enforce_offline_mode()
    model_lock = ModelLock.load(args.model_lock)

    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(
        str(model_lock.metric_path("pickscore_processor")),
        local_files_only=True,
    )
    model = CLIPModel.from_pretrained(
        str(model_lock.metric_path("pickscore_model")),
        local_files_only=True,
    ).eval().to(args.device)
    model = model.to(dtype=torch_dtype(args.dtype))

    for checkpoint in args.checkpoints:
        for dataset in args.datasets:
            records = existing_records(
                expected_records_for(checkpoint, dataset, manifest_dir, image_root, args.limit_prompts),
                args.limit_images,
            )
            out_dir = metric_output_dir(metric_root, "pickscore", checkpoint, dataset)
            per_image_path = out_dir / "per_image.jsonl"
            per_image_path.write_text("", encoding="utf-8")
            scores = []
            prompt_acc: dict[str, dict] = {}
            for batch in iter_batches(records, args.batch_size):
                for row, score in zip(batch, _score_batch(model, processor, batch, args.device, args.dtype)):
                    save_json_line(per_image_path, {**row, "pickscore": score})
                    scores.append(score)
                    if dataset in PER_PROMPT_DATASETS:
                        acc = prompt_acc.setdefault(
                            row["prompt_id"],
                            {
                                "checkpoint": checkpoint,
                                "dataset": dataset,
                                "prompt_id": row["prompt_id"],
                                "prompt": row["prompt"],
                                "sum": 0.0,
                                "count": 0,
                            },
                        )
                        acc["sum"] += score
                        acc["count"] += 1

            per_prompt = []
            if dataset in PER_PROMPT_DATASETS:
                per_prompt = [
                    {
                        "checkpoint": acc["checkpoint"],
                        "dataset": acc["dataset"],
                        "prompt_id": acc["prompt_id"],
                        "prompt": acc["prompt"],
                        "pickscore": acc["sum"] / acc["count"] if acc["count"] else None,
                        "count": acc["count"],
                    }
                    for acc in prompt_acc.values()
                ]
                write_jsonl(out_dir / "per_prompt.jsonl", per_prompt)
            summary = {"checkpoint": checkpoint, "dataset": dataset, "pickscore": summarize_scores(scores)}
            write_json(out_dir / "summary.json", summary)
            print(f"Wrote {out_dir}")
