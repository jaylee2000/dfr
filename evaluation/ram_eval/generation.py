from __future__ import annotations

import argparse
import gc
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from .config import (
    DATASETS,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_MODEL_LOCK,
    LORA_TARGET_MODULES,
)
from .io_utils import (
    ensure_dir,
    image_manifest_path,
    image_path,
    prompt_manifest_path,
    read_json,
    read_jsonl,
    seeds_path,
    write_json,
    write_jsonl,
)
from .model_assets import ModelLock, enforce_offline_mode
from .model_registry import CheckpointSpec, ModelSource, load_registry


def _checkpoint_payload(spec: CheckpointSpec) -> dict:
    payload = asdict(spec)
    source = payload.get("source")
    if source and source.get("path") is not None:
        source["path"] = str(source["path"])
    return payload


def _local_checkpoint_path(spec: CheckpointSpec) -> Path:
    assert spec.source is not None and spec.source.path is not None
    path = spec.source.path
    if spec.lora_subfolder:
        path = path / spec.lora_subfolder
    path = path.expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Configured local checkpoint path does not exist for {spec.key}: {path}")
    if spec.loader == "peft_lora":
        required = ("adapter_config.json", spec.lora_weight_name)
    elif spec.loader == "ram_full":
        required = ("model.safetensors",)
    else:
        required = ("model_index.json",)
    missing = [path / name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Configured local checkpoint {spec.key} is missing: {missing[0]}")
    return path


def resolve_checkpoint_specs(
    checkpoint_keys: Iterable[str],
    model_lock: ModelLock,
    catalog_paths: Iterable[str | Path] = (),
) -> list[CheckpointSpec]:
    registry = load_registry(tuple(catalog_paths))
    resolved = []
    for key in checkpoint_keys:
        spec = registry.checkpoint(key)
        if spec.loader == "base":
            path = model_lock.base_model_path(spec.base_model)
        elif spec.source is not None and spec.source.path is not None:
            path = _local_checkpoint_path(spec)
        else:
            path = model_lock.checkpoint_path(spec.key)
        weight_name = spec.lora_weight_name
        if spec.loader == "peft_lora" and not (path / weight_name).is_file():
            for candidate in ("adapter_model.safetensors", "adapter_model.bin"):
                if (path / candidate).is_file():
                    weight_name = candidate
                    break
        resolved.append(
            replace(
                spec,
                source=ModelSource(path=path),
                lora_subfolder=None,
                lora_weight_name=weight_name,
            )
        )
    return resolved


def _torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _normalize_state_keys(state: dict) -> dict:
    normalized = {}
    prefixes = ("module.", "_orig_mod.", "transformer.")
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        normalized[new_key] = value
    return normalized


def _load_ram_transformer(pipe, checkpoint_path: Path, device, dtype):
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=LORA_TARGET_MODULES,
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.add_adapter("old", lora_config)
    pipe.transformer.add_adapter("evaluation", lora_config)

    state = load_file(str(checkpoint_path / "model.safetensors"), device="cpu")
    state = _normalize_state_keys(state)
    missing, unexpected = pipe.transformer.load_state_dict(state, strict=False)
    pipe.transformer.set_adapter("evaluation")
    pipe.transformer.to(device=device, dtype=dtype)
    pipe.transformer.eval()
    del state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"RAM load_state_dict complete: {len(missing)} missing keys, "
        f"{len(unexpected)} unexpected keys."
    )
    if unexpected[:5]:
        print("  first unexpected:", unexpected[:5])
    if missing[:5]:
        print("  first missing:", missing[:5])
    return pipe


def load_pipeline(
    spec: CheckpointSpec,
    *,
    model_lock: ModelLock | None,
    device_name: str,
    dtype_name: str,
    attention_slicing: bool,
    cpu_offload: bool,
    merge_lora: bool,
):
    from diffusers import StableDiffusion3Pipeline
    from peft import PeftModel

    import torch

    dtype = _torch_dtype(dtype_name)
    device = torch.device(device_name)
    assert spec.source is not None and spec.source.path is not None
    checkpoint_path = spec.source.path
    if spec.loader == "base":
        base_model_path = checkpoint_path
    else:
        assert model_lock is not None
        base_model_path = model_lock.base_model_path(spec.base_model)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        str(base_model_path),
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    if spec.loader == "peft_lora":
        temporary_checkpoint = None
        peft_path = checkpoint_path
        if spec.lora_weight_name not in {"adapter_model.safetensors", "adapter_model.bin"}:
            temporary_checkpoint = TemporaryDirectory(prefix=f"ram-eval-{spec.key}-")
            peft_path = Path(temporary_checkpoint.name)
            (peft_path / "adapter_config.json").symlink_to(
                checkpoint_path / "adapter_config.json"
            )
            standard_name = (
                "adapter_model.safetensors"
                if Path(spec.lora_weight_name).suffix == ".safetensors"
                else "adapter_model.bin"
            )
            (peft_path / standard_name).symlink_to(checkpoint_path / spec.lora_weight_name)
        try:
            pipe.transformer = PeftModel.from_pretrained(
                pipe.transformer,
                str(peft_path),
                local_files_only=True,
            )
        finally:
            if temporary_checkpoint is not None:
                temporary_checkpoint.cleanup()
        if merge_lora:
            pipe.transformer = pipe.transformer.merge_and_unload()
        pipe.transformer.eval()
    elif spec.loader == "ram_full":
        pipe = _load_ram_transformer(pipe, checkpoint_path, device, dtype)
    elif spec.loader != "base":
        raise ValueError(f"Unsupported loader type for {spec.key}: {spec.loader}")

    if attention_slicing:
        pipe.enable_attention_slicing("auto")
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    return pipe


def _load_prompt_manifest(manifest_dir: Path, dataset: str) -> list[dict]:
    path = prompt_manifest_path(manifest_dir, dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {dataset} prompt manifest: {path}. Run python -m ram_eval.prompts first."
        )
    return list(read_jsonl(path))


def _load_seeds(manifest_dir: Path) -> list[int]:
    path = seeds_path(manifest_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing seed manifest: {path}. Run python -m ram_eval.prompts first.")
    return [int(seed) for seed in read_json(path)["seeds"]]


def _build_latents(pipe, seeds: Iterable[int], height: int, width: int):
    import torch

    latent_height = height // pipe.vae_scale_factor
    latent_width = width // pipe.vae_scale_factor
    latent_channels = pipe.transformer.config.in_channels
    latents_by_seed = {}
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        latents_by_seed[int(seed)] = torch.randn(
            (1, latent_channels, latent_height, latent_width),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
    return latents_by_seed


def expected_image_records(
    checkpoint_key: str,
    dataset: str,
    prompts: list[dict],
    seeds: list[int],
    image_root: Path,
) -> list[dict]:
    rows = []
    for prompt in prompts:
        for seed in seeds:
            path = image_path(image_root, checkpoint_key, dataset, prompt["prompt_id"], seed)
            rows.append(
                {
                    "checkpoint": checkpoint_key,
                    "dataset": dataset,
                    "prompt_id": prompt["prompt_id"],
                    "prompt_index": prompt["prompt_index"],
                    "source_index": prompt["source_index"],
                    "prompt": prompt["prompt"],
                    "seed": seed,
                    "image_path": str(path),
                    "exists": path.exists(),
                    "metadata": prompt.get("metadata", {}),
                    "real_image_path": prompt.get("real_image_path"),
                }
            )
    return rows


def _save_image_manifest(
    manifest_dir: Path,
    checkpoint_key: str,
    dataset: str,
    prompts: list[dict],
    seeds: list[int],
    image_root: Path,
) -> None:
    rows = expected_image_records(checkpoint_key, dataset, prompts, seeds, image_root)
    write_jsonl(image_manifest_path(manifest_dir, checkpoint_key, dataset), rows)


def _generate_dataset(pipe, spec: CheckpointSpec, dataset: str, prompts: list[dict], seeds: list[int], args):
    import torch

    image_root = Path(args.image_root)
    latents_by_seed = _build_latents(pipe, seeds, args.height, args.width)
    dtype = _torch_dtype(args.dtype)
    device = pipe._execution_device if hasattr(pipe, "_execution_device") else torch.device(args.device)
    generated = 0
    skipped = 0

    for prompt_record in prompts:
        prompt = prompt_record["prompt"]
        prompt_id = prompt_record["prompt_id"]
        pending = []
        for seed in seeds:
            out_path = image_path(image_root, spec.key, dataset, prompt_id, seed)
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            pending.append((seed, out_path))

        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            if not batch:
                continue
            batch_seeds = [seed for seed, _ in batch]
            batch_latents = torch.cat([latents_by_seed[seed].clone() for seed in batch_seeds], dim=0)
            batch_latents = batch_latents.to(device=device, dtype=dtype)
            with torch.inference_mode():
                outputs = pipe(
                    [prompt] * len(batch),
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    latents=batch_latents,
                )
            for image, (_, out_path) in zip(outputs.images, batch):
                ensure_dir(out_path.parent)
                image.save(out_path)
                generated += 1
            del outputs, batch_latents
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"{spec.key}/{dataset}: generated {generated}, skipped {skipped}")


def _run_checkpoint(spec: CheckpointSpec, args: argparse.Namespace, model_lock: ModelLock) -> None:
    checkpoint_args = argparse.Namespace(**vars(args))
    checkpoint_args.guidance_scale = (
        spec.guidance_scale if args.guidance_scale is None else args.guidance_scale
    )
    checkpoint_args.dtype = spec.default_dtype if args.dtype is None else args.dtype

    manifest_dir = Path(args.manifest_dir)
    image_root = Path(args.image_root)
    all_seeds = _load_seeds(manifest_dir)
    datasets = args.prompts

    dataset_payloads = []
    for dataset in datasets:
        prompts = _load_prompt_manifest(manifest_dir, dataset)
        if args.start_index:
            prompts = prompts[args.start_index :]
        if args.limit_prompts is not None:
            prompts = prompts[: args.limit_prompts]
        seeds = all_seeds[: args.images_per_prompt]
        if dataset == "mscoco":
            seeds = seeds[: args.mscoco_images_per_prompt]
        dataset_payloads.append((dataset, prompts, seeds))
        _save_image_manifest(manifest_dir, spec.key, dataset, prompts, seeds, image_root)

    run_config = {
        "checkpoint": _checkpoint_payload(spec),
        "prompts": datasets,
        "guidance_scale": checkpoint_args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
        "dtype": checkpoint_args.dtype,
        "batch_size": args.batch_size,
        "dry_run": args.dry_run,
    }
    write_json(Path(args.image_root) / spec.key / "generation_config.json", run_config)

    if args.dry_run:
        print(f"[dry-run] {spec.key}: loader={spec.loader}, dtype={checkpoint_args.dtype}")
        for dataset, prompts, seeds in dataset_payloads:
            print(f"[dry-run] {dataset}: {len(prompts)} prompts x {len(seeds)} seeds")
        return

    pipe = load_pipeline(
        spec,
        model_lock=model_lock,
        device_name=args.device,
        dtype_name=checkpoint_args.dtype,
        attention_slicing=args.attention_slicing,
        cpu_offload=args.cpu_offload,
        merge_lora=args.merge_lora,
    )
    for dataset, prompts, seeds in dataset_payloads:
        _generate_dataset(pipe, spec, dataset, prompts, seeds, checkpoint_args)
        _save_image_manifest(manifest_dir, spec.key, dataset, prompts, seeds, image_root)
    del pipe
    gc.collect()

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_generation(args: argparse.Namespace) -> None:
    model_lock = ModelLock.load(args.model_lock)
    specs = resolve_checkpoint_specs(args.checkpoints, model_lock, args.checkpoint_config)
    if not args.dry_run:
        enforce_offline_mode()
    for spec in specs:
        _run_checkpoint(spec, args, model_lock)


def build_generation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate images for registered checkpoints.")
    parser.set_defaults(
        manifest_dir=str(DEFAULT_MANIFEST_DIR),
        image_root=str(DEFAULT_IMAGE_ROOT),
        model_lock=str(DEFAULT_MODEL_LOCK),
    )
    parser.add_argument("--checkpoint-config", action="append", default=[])
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--prompts", nargs="+", choices=DATASETS, required=True)
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--images-per-prompt", type=int, default=64)
    parser.add_argument("--mscoco-images-per-prompt", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default=None,
        help="Override the inferred checkpoint dtype.",
    )
    parser.add_argument("--attention-slicing", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    run_generation(build_generation_parser().parse_args(argv))


if __name__ == "__main__":
    main()
