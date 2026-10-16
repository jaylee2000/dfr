from __future__ import annotations

from pathlib import Path

from .config import PER_PROMPT_DATASETS
from .model_assets import ModelLock, enforce_offline_mode
from .metrics_common import (
    existing_records,
    expected_records_for,
    group_by_prompt,
    iter_batches,
    metric_output_dir,
    open_rgb,
    summarize_scores,
    write_json,
    write_jsonl,
)


def _embed_rows(model, preprocess, rows: list[dict], batch_size: int, device: str):
    import torch

    embeddings = []
    for batch in iter_batches(rows, batch_size):
        tensors = []
        for row in batch:
            image = open_rgb(row["image_path"])
            try:
                tensor = preprocess(image)
            finally:
                image.close()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        images = torch.cat(tensors, dim=0).to(device)
        with torch.inference_mode():
            emb = model.embed(images).detach().cpu().float()
        embeddings.append(emb)
        del images, emb
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, 1)


def _mean_pairwise_cosine_distance(embeddings) -> float | None:
    import torch

    if embeddings.shape[0] < 2:
        return None
    if embeddings.ndim == 3 and embeddings.shape[1] > 1:
        cls_embedding = embeddings[:, 0, :]
        patch_embeddings = embeddings[:, 1:, :]
        patch_count = patch_embeddings.shape[1]
        patch_side = int(patch_count**0.5)
        if patch_side * patch_side == patch_count:
            patch_embedding = patch_embeddings.reshape(
                embeddings.shape[0],
                patch_side,
                patch_side,
                embeddings.shape[-1],
            ).mean(dim=(1, 2))
            embeddings = torch.cat((cls_embedding, patch_embedding), dim=-1)
        else:
            embeddings = embeddings.flatten(start_dim=1)
    elif embeddings.ndim != 2:
        embeddings = embeddings.flatten(start_dim=1)
    emb = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    sim = emb @ emb.T
    n = sim.shape[0]
    upper = sim[torch.triu_indices(n, n, offset=1).unbind()]
    return float((1.0 - upper).mean().item())


def run(args) -> None:
    metric_root = Path(args.metric_root)
    manifest_dir = Path(args.manifest_dir)
    image_root = Path(args.image_root)
    if args.dry_run:
        for checkpoint in args.checkpoints:
            for dataset in args.datasets:
                if dataset not in PER_PROMPT_DATASETS:
                    print(f"Skipping DreamSim diversity for {dataset}; it is defined for 64-image prompt groups.")
                    continue
                records = existing_records(
                    expected_records_for(checkpoint, dataset, manifest_dir, image_root, args.limit_prompts),
                    args.limit_images,
                )
                print(f"[dry-run] DreamSim {checkpoint}/{dataset}: {len(records)} existing images")
        return

    enforce_offline_mode()
    model_lock = ModelLock.load(args.model_lock)

    from dreamsim import dreamsim

    model, preprocess = dreamsim(
        dreamsim_type="dino_vitb16",
        use_patch_model=True,
        device=args.device,
        cache_dir=str(model_lock.metric_path("dreamsim")),
    )
    model.eval()

    for checkpoint in args.checkpoints:
        for dataset in args.datasets:
            if dataset not in PER_PROMPT_DATASETS:
                print(f"Skipping DreamSim diversity for {dataset}; it is defined for 64-image prompt groups.")
                continue
            records = existing_records(
                expected_records_for(checkpoint, dataset, manifest_dir, image_root, args.limit_prompts),
                args.limit_images,
            )
            out_dir = metric_output_dir(metric_root, "dreamsim", checkpoint, dataset)

            per_prompt = []
            for prompt_id, rows in group_by_prompt(records).items():
                embeddings = _embed_rows(model, preprocess, rows, args.batch_size, args.device)
                score = _mean_pairwise_cosine_distance(embeddings)
                per_prompt.append(
                    {
                        "checkpoint": checkpoint,
                        "dataset": dataset,
                        "prompt_id": prompt_id,
                        "prompt": rows[0]["prompt"],
                        "dreamsim_diversity": score,
                        "count": len(rows),
                    }
                )

            scores = [row["dreamsim_diversity"] for row in per_prompt if row["dreamsim_diversity"] is not None]
            write_jsonl(out_dir / "per_prompt.jsonl", per_prompt)
            write_json(out_dir / "summary.json", {"checkpoint": checkpoint, "dataset": dataset, "dreamsim_diversity": summarize_scores(scores)})
            print(f"Wrote {out_dir}")
