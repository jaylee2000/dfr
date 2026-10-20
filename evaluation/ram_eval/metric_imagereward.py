from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from .config import PER_PROMPT_DATASETS
from .model_assets import ModelLock, enforce_offline_mode
from .metrics_common import (
    existing_records,
    expected_records_for,
    group_by_prompt,
    iter_batches,
    metric_output_dir,
    normalize_score_output,
    open_rgb,
    save_json_line,
    summarize_scores,
    torch_dtype,
    write_json,
    write_jsonl,
)


def _build_local_tokenizer(tokenizer_path: Path):
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
    )
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def _load_model(image_reward_root: Path, tokenizer_path: Path):
    from reward_models.image_reward import rm_load
    from ImageReward.models.BLIP import blip_pretrain

    blip_pretrain.init_tokenizer = lambda: _build_local_tokenizer(tokenizer_path)
    return rm_load(
        name=str(image_reward_root / "ImageReward.pt"),
        med_config=str(image_reward_root / "med_config.json"),
    )


def _score_batch(model, prompt: str, rows: list[dict], device: str, dtype_name: str) -> list[float]:
    import torch

    images = []
    dtype = torch_dtype(dtype_name)
    device_type = device.split(":", 1)[0]
    use_autocast = device_type == "cuda" and dtype != torch.float32
    autocast_context = torch.autocast(device_type=device_type, dtype=dtype) if use_autocast else nullcontext()
    try:
        images = [open_rgb(row["image_path"]) for row in rows]
        with torch.inference_mode(), autocast_context:
            _rank, rewards = model.inference_rank(prompt, images)
        scores = normalize_score_output(rewards)
    finally:
        for image in images:
            image.close()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if len(scores) == 1 and len(rows) > 1:
        scores = scores * len(rows)
    if len(scores) != len(rows):
        raise RuntimeError(f"ImageReward returned {len(scores)} scores for {len(rows)} images.")
    return scores


def _score_group(model, prompt: str, rows: list[dict], batch_size: int, device: str, dtype_name: str) -> list[float]:
    scores = []
    for batch in iter_batches(rows, batch_size):
        scores.extend(_score_batch(model, prompt, batch, device, dtype_name))
    return scores


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
                print(f"[dry-run] ImageReward {checkpoint}/{dataset}: {len(records)} existing images")
        return

    enforce_offline_mode()
    model_lock = ModelLock.load(args.model_lock)
    image_reward_root = model_lock.metric_path("imagereward")
    tokenizer_path = model_lock.metric_path("imagereward_tokenizer")

    import torch

    model = _load_model(image_reward_root, tokenizer_path).eval().to(
        device=args.device,
        dtype=torch_dtype(args.dtype),
    )

    for checkpoint in args.checkpoints:
        for dataset in args.datasets:
            records = existing_records(
                expected_records_for(checkpoint, dataset, manifest_dir, image_root, args.limit_prompts),
                args.limit_images,
            )
            out_dir = metric_output_dir(metric_root, "imagereward", checkpoint, dataset)
            per_image_path = out_dir / "per_image.jsonl"
            per_image_path.write_text("", encoding="utf-8")

            scores = []
            prompt_acc: dict[str, dict] = {}
            grouped = group_by_prompt(records)
            for _prompt_id, rows in grouped.items():
                group_scores = _score_group(
                    model,
                    rows[0]["prompt"],
                    rows,
                    args.batch_size,
                    args.device,
                    args.dtype,
                )
                for row, score in zip(rows, group_scores):
                    save_json_line(per_image_path, {**row, "imagereward": score})
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
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            per_prompt = []
            if dataset in PER_PROMPT_DATASETS:
                per_prompt = [
                    {
                        "checkpoint": acc["checkpoint"],
                        "dataset": acc["dataset"],
                        "prompt_id": acc["prompt_id"],
                        "prompt": acc["prompt"],
                        "imagereward": acc["sum"] / acc["count"] if acc["count"] else None,
                        "count": acc["count"],
                    }
                    for acc in prompt_acc.values()
                ]
                write_jsonl(out_dir / "per_prompt.jsonl", per_prompt)
            summary = {
                "checkpoint": checkpoint,
                "dataset": dataset,
                "imagereward": summarize_scores(scores),
            }
            write_json(out_dir / "summary.json", summary)
            print(f"Wrote {out_dir}")
