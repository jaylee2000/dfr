from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from .config import DEFAULT_MODEL_LOCK, DEFAULT_MODEL_ROOT
from .metric_dreamsim import DREAMSIM_REQUIRED_FILES, DREAMSIM_VARIANTS
from .model_assets import LOCK_VERSION, ModelAssetError, ModelLock, verify_lock
from .model_registry import BaseModelSpec, CheckpointSpec, ModelRegistry, ModelSource, load_registry


METRIC_ASSETS = ("dreamsim", "fid", "imagereward", "pickscore")


def _remote_snapshot(
    source: ModelSource,
    destination: Path,
    *,
    allow_patterns: Sequence[str] | None = None,
) -> tuple[Path, str]:
    from huggingface_hub import HfApi, snapshot_download

    assert source.repo_id is not None
    requested_revision = source.revision or "main"
    resolved_revision = HfApi().model_info(source.repo_id, revision=requested_revision).sha
    destination = destination / resolved_revision
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=source.repo_id,
        revision=resolved_revision,
        local_dir=destination,
        allow_patterns=allow_patterns,
    )
    return destination.resolve(), resolved_revision


def _resolve_source(source: ModelSource, destination: Path) -> tuple[Path, dict[str, str]]:
    if source.path is not None:
        path = source.path.resolve()
        if not path.exists():
            raise ModelAssetError(f"Configured local model path does not exist: {path}")
        return path, {"path": str(path)}
    path, resolved_revision = _remote_snapshot(source, destination)
    return path, {
        "repo_id": str(source.repo_id),
        "requested_revision": str(source.revision or "main"),
        "resolved_revision": resolved_revision,
    }


def _require_files(label: str, root: Path, filenames: Sequence[str]) -> None:
    missing = [str(root / filename) for filename in filenames if not (root / filename).is_file()]
    if missing:
        raise ModelAssetError(f"{label} is incomplete; missing: {', '.join(missing)}")


def _prepare_base_model(spec: BaseModelSpec, model_root: Path) -> dict[str, Any]:
    path, source = _resolve_source(spec.source, model_root / "snapshots" / "base_models" / spec.key)
    _require_files(f"Base model {spec.key}", path, ("model_index.json",))
    return {"path": str(path), "required": ["model_index.json"], "source": source}


def _prepare_checkpoint(spec: CheckpointSpec, model_root: Path, base_entry: dict[str, Any]) -> dict[str, Any]:
    if spec.loader == "base":
        return {
            "path": base_entry["path"],
            "required": ["model_index.json"],
            "loader": spec.loader,
        }

    assert spec.source is not None
    source_root, source = _resolve_source(
        spec.source,
        model_root / "snapshots" / "checkpoints" / spec.key,
    )
    checkpoint_root = source_root / spec.lora_subfolder if spec.lora_subfolder else source_root
    if spec.loader == "ram_full":
        _require_files(f"Checkpoint {spec.key}", checkpoint_root, ("model.safetensors",))
        required = ["model.safetensors"]
    elif spec.loader == "peft_lora":
        _require_files(
            f"Checkpoint {spec.key}",
            checkpoint_root,
            ("adapter_config.json", spec.lora_weight_name),
        )
        if spec.lora_weight_name != "adapter_model.safetensors":
            normalized_root = model_root / "prepared_checkpoints" / spec.key
            normalized_root.mkdir(parents=True, exist_ok=True)
            normalized_weight_name = (
                "adapter_model.safetensors"
                if Path(spec.lora_weight_name).suffix == ".safetensors"
                else "adapter_model.bin"
            )
            for stale_name in ("adapter_model.safetensors", "adapter_model.bin"):
                stale_path = normalized_root / stale_name
                if stale_path.exists() and stale_name != normalized_weight_name:
                    stale_path.unlink()
            shutil.copy2(checkpoint_root / "adapter_config.json", normalized_root / "adapter_config.json")
            shutil.copy2(checkpoint_root / spec.lora_weight_name, normalized_root / normalized_weight_name)
            checkpoint_root = normalized_root.resolve()
            required = ["adapter_config.json", normalized_weight_name]
        else:
            required = ["adapter_config.json", "adapter_model.safetensors"]
    else:
        raise ModelAssetError(f"Unsupported loader for {spec.key}: {spec.loader}")
    return {
        "path": str(checkpoint_root.resolve()),
        "required": required,
        "loader": spec.loader,
        "source": source,
    }


def _prepare_hf_metric(repo_id: str, destination: Path) -> dict[str, Any]:
    path, revision = _remote_snapshot(ModelSource(repo_id=repo_id, revision="main"), destination)
    return {
        "path": str(path),
        "required": ["config.json"],
        "source": {"repo_id": repo_id, "requested_revision": "main", "resolved_revision": revision},
    }


def _prepare_dreamsim(model_root: Path) -> dict[str, Any]:
    from dreamsim import dreamsim

    path = (model_root / "metrics" / "dreamsim").resolve()
    path.mkdir(parents=True, exist_ok=True)
    for dreamsim_type, use_patch_model in DREAMSIM_VARIANTS.values():
        model, preprocess = dreamsim(
            dreamsim_type=dreamsim_type,
            use_patch_model=use_patch_model,
            device="cpu",
            cache_dir=str(path),
        )
        del model, preprocess
    required = list(DREAMSIM_REQUIRED_FILES)
    _require_files("DreamSim", path, required)
    return {"path": str(path), "required": required}


def _prepare_inception(model_root: Path) -> dict[str, Any]:
    import torch
    from torchvision.models import Inception_V3_Weights

    path = (model_root / "metrics" / "inception").resolve()
    path.mkdir(parents=True, exist_ok=True)
    filename = Path(Inception_V3_Weights.IMAGENET1K_V1.url).name
    target = path / filename
    if not target.is_file():
        torch.hub.download_url_to_file(
            Inception_V3_Weights.IMAGENET1K_V1.url,
            str(target),
            hash_prefix="0cc3c7bd",
            progress=True,
        )
    return {"path": str(path), "required": [filename]}


def _prepare_dinov2(model_root: Path) -> dict[str, Any]:
    path, revision = _remote_snapshot(
        ModelSource(repo_id="facebook/dinov2-large", revision="main"),
        model_root / "metrics" / "dinov2_vitl14",
        allow_patterns=["config.json", "model.safetensors"],
    )
    required = ["config.json", "model.safetensors"]
    _require_files("DINOv2 ViT-L/14", path, required)
    return {
        "path": str(path),
        "required": required,
        "source": {"repo_id": "facebook/dinov2-large", "requested_revision": "main", "resolved_revision": revision},
    }


def verify_metric_assets(lock: ModelLock) -> list[str]:
    errors = []
    requirements = {
        "inception_v3": ("inception_v3_google-0cc3c7bd.pth",),
        "dinov2_vitl14": ("config.json", "model.safetensors"),
        "dreamsim": DREAMSIM_REQUIRED_FILES,
        "imagereward": ("ImageReward.pt", "med_config.json"),
        "imagereward_tokenizer": ("tokenizer_config.json", "vocab.txt"),
        "pickscore_processor": ("config.json",),
        "pickscore_model": ("config.json",),
    }
    for name, required in requirements.items():
        try:
            lock.metric_path(name, required=required)
        except ModelAssetError as exc:
            errors.append(str(exc))
    return errors


def _prepare_imagereward(model_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    image_reward_repo = "THUDM/ImageReward"
    path, revision = _remote_snapshot(
        ModelSource(repo_id=image_reward_repo, revision="main"),
        model_root / "metrics" / "imagereward",
        allow_patterns=["ImageReward.pt", "med_config.json"],
    )
    required = ["ImageReward.pt", "med_config.json"]
    _require_files("ImageReward", path, required)
    image_reward_entry = {
        "path": str(path),
        "required": required,
        "source": {
            "repo_id": image_reward_repo,
            "requested_revision": "main",
            "resolved_revision": revision,
        },
    }

    tokenizer_repo = "google-bert/bert-base-uncased"
    tokenizer_path, tokenizer_revision = _remote_snapshot(
        ModelSource(repo_id=tokenizer_repo, revision="main"),
        model_root / "metrics" / "imagereward_tokenizer",
        allow_patterns=["*.json", "vocab.txt"],
    )
    tokenizer_required = ["tokenizer_config.json", "vocab.txt"]
    _require_files("ImageReward tokenizer", tokenizer_path, tokenizer_required)
    tokenizer_entry = {
        "path": str(tokenizer_path),
        "required": tokenizer_required,
        "source": {
            "repo_id": tokenizer_repo,
            "requested_revision": "main",
            "resolved_revision": tokenizer_revision,
        },
    }
    return image_reward_entry, tokenizer_entry


def _prepare_metrics(metric_names: Sequence[str], model_root: Path) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for metric in metric_names:
        print(f"Preparing metric assets: {metric}", flush=True)
        if metric == "dreamsim":
            entries["dreamsim"] = _prepare_dreamsim(model_root)
        elif metric == "fid":
            entries["inception_v3"] = _prepare_inception(model_root)
            entries["dinov2_vitl14"] = _prepare_dinov2(model_root)
        elif metric == "imagereward":
            image_reward_entry, tokenizer_entry = _prepare_imagereward(model_root)
            entries["imagereward"] = image_reward_entry
            entries["imagereward_tokenizer"] = tokenizer_entry
        elif metric == "pickscore":
            entries["pickscore_processor"] = _prepare_hf_metric(
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                model_root / "metrics" / "pickscore_processor",
            )
            entries["pickscore_model"] = _prepare_hf_metric(
                "yuvalkirstain/PickScore_v1",
                model_root / "metrics" / "pickscore_model",
            )
        else:
            raise ModelAssetError(f"Unsupported metric: {metric}")
    return entries


def prepare_models(
    registry: ModelRegistry,
    checkpoint_keys: Sequence[str],
    metric_names: Sequence[str],
    model_root: Path,
    lock_path: Path,
) -> ModelLock:
    model_root = model_root.expanduser().resolve()
    lock_path = lock_path.expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    selected = [registry.checkpoint(key) for key in checkpoint_keys]
    base_keys = tuple(dict.fromkeys(spec.base_model for spec in selected))

    base_entries: dict[str, Any] = {}
    for key in base_keys:
        print(f"Preparing base model: {key}", flush=True)
        base_entries[key] = _prepare_base_model(registry.base_models[key], model_root)

    checkpoint_entries: dict[str, Any] = {}
    for spec in selected:
        print(f"Preparing checkpoint: {spec.key}", flush=True)
        checkpoint_entries[spec.key] = _prepare_checkpoint(spec, model_root, base_entries[spec.base_model])

    payload = {
        "version": LOCK_VERSION,
        "base_models": base_entries,
        "checkpoints": checkpoint_entries,
        "metrics": _prepare_metrics(metric_names, model_root),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = lock_path.with_name(f"{lock_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(lock_path)
    lock = ModelLock.load(lock_path)
    errors = verify_lock(lock)
    if errors:
        raise ModelAssetError("Prepared model lock failed verification:\n" + "\n".join(errors))
    return lock


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and verify offline model assets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "prepare",
        help="Prepare every bundled checkpoint and metric and write the one-time lock file.",
    )

    subparsers.add_parser("verify", help="Verify that every locked artifact exists locally.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        registry = load_registry(include_local=False)
        lock = prepare_models(
            registry,
            list(registry.checkpoints),
            list(METRIC_ASSETS),
            DEFAULT_MODEL_ROOT,
            DEFAULT_MODEL_LOCK,
        )
        print(f"Prepared and verified model lock: {lock.path}")
        return 0

    lock = ModelLock.load(DEFAULT_MODEL_LOCK)
    errors = verify_lock(lock) + verify_metric_assets(lock)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"Verified model lock: {lock.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
