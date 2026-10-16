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
    torch_dtype,
    write_json,
    write_jsonl,
)


class OnlineFeatureStats:
    def __init__(self, dim: int):
        import numpy as np

        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros((dim, dim), dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        import numpy as np

        batch = np.asarray(batch, dtype=np.float64)
        for x in batch:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.m2 += np.outer(delta, delta2)

    def covariance(self) -> np.ndarray:
        import numpy as np

        if self.n < 2:
            return np.eye(self.mean.shape[0], dtype=np.float64) * 1e-6
        return self.m2 / (self.n - 1)

    def save(self, path: Path) -> None:
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, n=self.n, mean=self.mean, cov=self.covariance())

    @classmethod
    def load(cls, path: Path) -> "OnlineFeatureStats":
        import numpy as np

        data = np.load(path)
        stats = cls(int(data["mean"].shape[0]))
        stats.n = int(data["n"])
        stats.mean = data["mean"].astype(np.float64)
        cov = data["cov"].astype(np.float64)
        stats.m2 = cov * max(stats.n - 1, 1)
        return stats


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    import numpy as np

    vals, vecs = np.linalg.eigh((matrix + matrix.T) * 0.5)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance(stats_a: OnlineFeatureStats, stats_b: OnlineFeatureStats) -> float:
    import numpy as np

    mu1, mu2 = stats_a.mean, stats_b.mean
    cov1, cov2 = stats_a.covariance(), stats_b.covariance()
    diff = mu1 - mu2
    cov1_sqrt = _sqrtm_psd(cov1)
    covmean = _sqrtm_psd(cov1_sqrt @ cov2 @ cov1_sqrt)
    value = diff.dot(diff) + np.trace(cov1 + cov2 - 2.0 * covmean)
    return float(np.real(value))


def _build_inception(device: str, weights_path: Path):
    import torch
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(
        weights=None,
        aux_logits=True,
        transform_input=True,
        init_weights=False,
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, weights.transforms()


def _extract_features(model, transform, rows: list[dict], batch_size: int, device: str) -> np.ndarray:
    import numpy as np
    import torch

    chunks = []
    for batch in iter_batches(rows, batch_size):
        images = [transform(open_rgb(row["image_path"])) for row in batch]
        tensor = torch.stack(images, dim=0).to(device)
        with torch.inference_mode():
            feats = model(tensor)
        if isinstance(feats, tuple):
            feats = feats[0]
        chunks.append(feats.detach().cpu().float().numpy())
        del tensor, feats
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if not chunks:
        return np.empty((0, 2048), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _stats_from_rows(model, transform, rows: list[dict], batch_size: int, device: str) -> OnlineFeatureStats:
    stats = OnlineFeatureStats(2048)
    for batch in iter_batches(rows, batch_size):
        stats.update(_extract_features(model, transform, batch, batch_size, device))
    return stats


def _real_coco_rows(records: list[dict]) -> list[dict]:
    rows = []
    for row in records:
        real_path = row.get("real_image_path")
        if real_path and Path(real_path).exists():
            copied = dict(row)
            copied["image_path"] = real_path
            rows.append(copied)
    return rows


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
                )
                print(f"[dry-run] FID/FD {checkpoint}/{dataset}: {len(records)} existing images")
        return

    enforce_offline_mode()
    model_lock = ModelLock.load(args.model_lock)
    inception_root = model_lock.metric_path("inception_v3")
    model, transform = _build_inception(
        args.device,
        inception_root / "inception_v3_google-0cc3c7bd.pth",
    )

    for checkpoint in args.checkpoints:
        for dataset in args.datasets:
            records = existing_records(
                expected_records_for(checkpoint, dataset, manifest_dir, image_root, args.limit_prompts),
                args.limit_images,
            )
            out_dir = metric_output_dir(metric_root, "fid", checkpoint, dataset)

            if dataset in PER_PROMPT_DATASETS:
                base_records = existing_records(
                    expected_records_for(args.base_checkpoint, dataset, manifest_dir, image_root, args.limit_prompts),
                    args.limit_images,
                )
                generated_by_prompt = group_by_prompt(records)
                base_by_prompt = group_by_prompt(base_records)
                per_prompt = []
                for prompt_id, rows in generated_by_prompt.items():
                    ref_rows = base_by_prompt.get(prompt_id, [])
                    if len(rows) < 2 or len(ref_rows) < 2:
                        fd = None
                    else:
                        stats_gen = _stats_from_rows(model, transform, rows, args.batch_size, args.device)
                        stats_ref = _stats_from_rows(model, transform, ref_rows, args.batch_size, args.device)
                        fd = frechet_distance(stats_gen, stats_ref)
                    per_prompt.append(
                        {
                            "checkpoint": checkpoint,
                            "dataset": dataset,
                            "prompt_id": prompt_id,
                            "prompt": rows[0]["prompt"],
                            "fd_to_base": fd,
                            "count": len(rows),
                            "base_count": len(ref_rows),
                        }
                    )
                scores = [row["fd_to_base"] for row in per_prompt if row["fd_to_base"] is not None]
                write_jsonl(out_dir / "per_prompt.jsonl", per_prompt)
                write_json(out_dir / "summary.json", {"checkpoint": checkpoint, "dataset": dataset, "fd_to_base": summarize_scores(scores)})
                print(f"Wrote {out_dir}")
            elif dataset == "mscoco":
                real_rows = _real_coco_rows(
                    expected_records_for(checkpoint, dataset, manifest_dir, image_root, args.limit_prompts)
                )
                if args.limit_images is not None:
                    real_rows = real_rows[: args.limit_images]
                if not records or not real_rows:
                    fid = None
                else:
                    stats_gen = _stats_from_rows(model, transform, records, args.batch_size, args.device)
                    stats_real = _stats_from_rows(model, transform, real_rows, args.batch_size, args.device)
                    fid = frechet_distance(stats_gen, stats_real)
                    stats_gen.save(out_dir / "generated_stats.npz")
                    stats_real.save(out_dir / "real_stats.npz")
                write_json(out_dir / "summary.json", {"checkpoint": checkpoint, "dataset": dataset, "fid_30k": fid, "generated_count": len(records), "real_count": len(real_rows)})
                print(f"Wrote {out_dir}")
