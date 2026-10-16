from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_CATALOG = ROOT / "configs" / "checkpoints.yaml"
DEFAULT_LOCAL_CHECKPOINT_CATALOG = ROOT / "configs" / "checkpoints.local.yaml"
LOADERS = ("base", "peft_lora", "ram_full")
DTYPES = ("fp16", "bf16", "fp32")
KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSource:
    repo_id: str | None = None
    revision: str | None = None
    path: Path | None = None

    @property
    def is_remote(self) -> bool:
        return self.repo_id is not None


@dataclass(frozen=True)
class BaseModelSpec:
    key: str
    display_name: str
    source: ModelSource


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    display_name: str
    base_model: str
    source: ModelSource | None
    loader: str
    guidance_scale: float
    default_dtype: str
    lora_subfolder: str | None = None
    lora_weight_name: str = "adapter_model.safetensors"


@dataclass(frozen=True)
class ModelRegistry:
    base_models: Mapping[str, BaseModelSpec]
    checkpoints: Mapping[str, CheckpointSpec]

    def checkpoint(self, key: str) -> CheckpointSpec:
        try:
            return self.checkpoints[key]
        except KeyError as exc:
            choices = ", ".join(self.checkpoints)
            raise RegistryError(f"Unknown checkpoint {key!r}. Available checkpoints: {choices}") from exc


def default_catalog_paths(
    extra_paths: Sequence[str | Path] = (), *, include_local: bool = True
) -> tuple[Path, ...]:
    paths = [DEFAULT_CHECKPOINT_CATALOG]
    if include_local and DEFAULT_LOCAL_CHECKPOINT_CATALOG.exists():
        paths.append(DEFAULT_LOCAL_CHECKPOINT_CATALOG)
    for raw_path in extra_paths:
        path = Path(raw_path).expanduser().resolve()
        if path not in paths:
            paths.append(path)
    return tuple(paths)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError(f"{context} must be a mapping")
    return value


def _required_string(data: Mapping[str, Any], name: str, context: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{context}.{name} must be a non-empty string")
    return value.strip()


def _validate_key(key: Any, context: str) -> str:
    if not isinstance(key, str) or not KEY_PATTERN.fullmatch(key):
        raise RegistryError(
            f"{context} keys must start with an alphanumeric character and contain only letters, numbers, ., _, or -"
        )
    return key


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(data).difference(allowed)
    if unknown:
        raise RegistryError(f"{context} has unknown fields: {', '.join(sorted(unknown))}")


def _source(value: Any, catalog_path: Path, context: str) -> ModelSource:
    data = _mapping(value, context)
    _reject_unknown(data, {"repo_id", "revision", "path"}, context)
    repo_id = data.get("repo_id")
    raw_path = data.get("path")
    if bool(repo_id) == bool(raw_path):
        raise RegistryError(f"{context} must define exactly one of repo_id or path")
    if repo_id:
        if not isinstance(repo_id, str):
            raise RegistryError(f"{context}.repo_id must be a string")
        revision = data.get("revision", "main")
        if not isinstance(revision, str) or not revision.strip():
            raise RegistryError(f"{context}.revision must be a non-empty string")
        return ModelSource(repo_id=repo_id.strip(), revision=revision.strip())
    if not isinstance(raw_path, str):
        raise RegistryError(f"{context}.path must be a string")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = catalog_path.parent / path
    return ModelSource(path=path.resolve())


def _load_catalog(path: Path) -> tuple[dict[str, BaseModelSpec], dict[str, CheckpointSpec]]:
    if not path.is_file():
        raise RegistryError(f"Checkpoint catalog does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    data = _mapping(raw, str(path))
    _reject_unknown(data, {"version", "base_models", "checkpoints"}, str(path))
    if data.get("version") != 1:
        raise RegistryError(f"{path}: version must be 1")

    base_models: dict[str, BaseModelSpec] = {}
    for key, raw_spec in _mapping(data.get("base_models", {}), f"{path}: base_models").items():
        context = f"{path}: base_models.{key}"
        key = _validate_key(key, f"{path}: base model")
        spec = _mapping(raw_spec, context)
        _reject_unknown(spec, {"display_name", "source"}, context)
        base_models[key] = BaseModelSpec(
            key=key,
            display_name=_required_string(spec, "display_name", context),
            source=_source(spec.get("source"), path, f"{context}.source"),
        )

    checkpoints: dict[str, CheckpointSpec] = {}
    for key, raw_spec in _mapping(data.get("checkpoints", {}), f"{path}: checkpoints").items():
        context = f"{path}: checkpoints.{key}"
        key = _validate_key(key, f"{path}: checkpoint")
        spec = _mapping(raw_spec, context)
        _reject_unknown(
            spec,
            {
                "display_name",
                "base_model",
                "source",
                "loader",
                "guidance_scale",
                "default_dtype",
                "lora_subfolder",
                "lora_weight_name",
            },
            context,
        )
        loader = _required_string(spec, "loader", context)
        if loader not in LOADERS:
            raise RegistryError(f"{context}.loader must be one of {', '.join(LOADERS)}")
        dtype = _required_string(spec, "default_dtype", context)
        if dtype not in DTYPES:
            raise RegistryError(f"{context}.default_dtype must be one of {', '.join(DTYPES)}")
        source_value = spec.get("source")
        source = _source(source_value, path, f"{context}.source") if source_value is not None else None
        if loader == "base" and source is not None:
            raise RegistryError(f"{context}.source must be omitted for the base loader")
        if loader != "base" and source is None:
            raise RegistryError(f"{context}.source is required for loader={loader}")
        try:
            guidance_scale = float(spec["guidance_scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError(f"{context}.guidance_scale must be numeric") from exc
        lora_subfolder = spec.get("lora_subfolder")
        if lora_subfolder is not None and (not isinstance(lora_subfolder, str) or not lora_subfolder):
            raise RegistryError(f"{context}.lora_subfolder must be a non-empty string when set")
        if lora_subfolder is not None:
            subfolder_path = Path(lora_subfolder)
            if subfolder_path.is_absolute() or ".." in subfolder_path.parts:
                raise RegistryError(f"{context}.lora_subfolder must stay within the model source")
        lora_weight_name = spec.get("lora_weight_name", "adapter_model.safetensors")
        if not isinstance(lora_weight_name, str) or not lora_weight_name:
            raise RegistryError(f"{context}.lora_weight_name must be a non-empty string")
        if Path(lora_weight_name).name != lora_weight_name:
            raise RegistryError(f"{context}.lora_weight_name must be a filename, not a path")
        if Path(lora_weight_name).suffix not in {".safetensors", ".bin"}:
            raise RegistryError(f"{context}.lora_weight_name must end in .safetensors or .bin")
        checkpoints[key] = CheckpointSpec(
            key=key,
            display_name=_required_string(spec, "display_name", context),
            base_model=_required_string(spec, "base_model", context),
            source=source,
            loader=loader,
            guidance_scale=guidance_scale,
            default_dtype=dtype,
            lora_subfolder=lora_subfolder,
            lora_weight_name=lora_weight_name,
        )
    return base_models, checkpoints


def load_registry(
    catalog_paths: Sequence[str | Path] = (), *, include_local: bool = True
) -> ModelRegistry:
    paths = default_catalog_paths(catalog_paths, include_local=include_local)
    base_models: dict[str, BaseModelSpec] = {}
    checkpoints: dict[str, CheckpointSpec] = {}
    for path in paths:
        catalog_bases, catalog_checkpoints = _load_catalog(path)
        duplicate_bases = set(base_models).intersection(catalog_bases)
        duplicate_checkpoints = set(checkpoints).intersection(catalog_checkpoints)
        if duplicate_bases:
            raise RegistryError(f"Duplicate base model keys: {', '.join(sorted(duplicate_bases))}")
        if duplicate_checkpoints:
            raise RegistryError(f"Duplicate checkpoint keys: {', '.join(sorted(duplicate_checkpoints))}")
        base_models.update(catalog_bases)
        checkpoints.update(catalog_checkpoints)

    for checkpoint in checkpoints.values():
        if checkpoint.base_model not in base_models:
            raise RegistryError(
                f"Checkpoint {checkpoint.key!r} references unknown base model {checkpoint.base_model!r}"
            )
    return ModelRegistry(base_models=base_models, checkpoints=checkpoints)
