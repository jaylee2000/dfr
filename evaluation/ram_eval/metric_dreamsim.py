from __future__ import annotations

from pathlib import Path

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


DREAMSIM_VARIANTS = {
    "dino_vitb16_patch": ("dino_vitb16", True),
    "open_clip_vitb32": ("open_clip_vitb32", False),
}
DREAMSIM_REQUIRED_FILES = (
    "dino_vitb16_pretrain.pth",
    "dino_vitb16_patch_lora/adapter_config.json",
    "dino_vitb16_patch_lora/adapter_model.safetensors",
    "facebookresearch_dino_main/hubconf.py",
    "open_clip_vitb32_pretrain.pth.tar",
    "open_clip_vitb32_single_lora/adapter_config.json",
    "open_clip_vitb32_single_lora/adapter_model.safetensors",
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
    upper = sim[torch.triu_indices(n, n, offset=1, device=sim.device).unbind()]
    return float((1.0 - upper).mean().item())


def run(args) -> None:
    metric_root = Path(args.metric_root)
    manifest_dir = Path(args.manifest_dir)
    image_root = Path(args.image_root)
    workloads = {}
    outputs = {}
    for checkpoint in args.checkpoints:
        for dataset in args.datasets:
            records = existing_records(
                expected_records_for(checkpoint, dataset, manifest_dir, image_root, args.limit_prompts),
                args.limit_images,
                allow_missing=args.dry_run,
            )
            if args.dry_run:
                print(f"[dry-run] DreamSim {checkpoint}/{dataset}: {len(records)} existing images")
                continue
            key = (checkpoint, dataset)
            workloads[key] = group_by_prompt(records)
            outputs[key] = {
                prompt_id: {
                    "checkpoint": checkpoint,
                    "dataset": dataset,
                    "prompt_id": prompt_id,
                    "prompt_index": rows[0]["prompt_index"],
                    "prompt": rows[0]["prompt"],
                    "count": len(rows),
                }
                for prompt_id, rows in workloads[key].items()
            }
    if args.dry_run:
        return

    enforce_offline_mode()
    model_lock = ModelLock.load(args.model_lock)
    cache_dir = model_lock.metric_path("dreamsim", required=DREAMSIM_REQUIRED_FILES)

    from dreamsim import dreamsim

    for variant, (dreamsim_type, use_patch_model) in DREAMSIM_VARIANTS.items():
        model, preprocess = dreamsim(
            dreamsim_type=dreamsim_type,
            use_patch_model=use_patch_model,
            device=args.device,
            cache_dir=str(cache_dir),
        )
        model.eval()
        for key, groups in workloads.items():
            for prompt_id, rows in groups.items():
                embeddings = _embed_rows(model, preprocess, rows, args.batch_size, args.device)
                outputs[key][prompt_id][variant] = _mean_pairwise_cosine_distance(embeddings)
        del model, preprocess
        if args.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    for (checkpoint, dataset), by_prompt in outputs.items():
        per_prompt = list(by_prompt.values())
        summaries = {
            variant: summarize_scores([row[variant] for row in per_prompt])
            for variant in DREAMSIM_VARIANTS
        }
        out = metric_output_dir(metric_root, "dreamsim", checkpoint, dataset)
        write_jsonl(out / "per_prompt.jsonl", per_prompt)
        write_json(out / "summary.json", {
            "checkpoint": checkpoint,
            "dataset": dataset,
            "dreamsim_diversity": summaries,
        })
        print(f"Wrote {out}")
