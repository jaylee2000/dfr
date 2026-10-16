from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


LOCK_VERSION = 1
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


class ModelAssetError(RuntimeError):
    pass


def enforce_offline_mode() -> None:
    for name, value in OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value


@dataclass(frozen=True)
class ModelLock:
    path: Path
    payload: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ModelLock":
        lock_path = Path(path).expanduser().resolve()
        if not lock_path.is_file():
            raise ModelAssetError(
                f"Model lock does not exist: {lock_path}. Prepare it on an internet-connected node with "
                "scripts/prepare_models.sh prepare."
            )
        with lock_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != LOCK_VERSION:
            raise ModelAssetError(f"Unsupported model lock version in {lock_path}")
        return cls(path=lock_path, payload=payload)

    def _entry(self, section: str, key: str) -> Mapping[str, Any]:
        entries = self.payload.get(section, {})
        try:
            entry = entries[key]
        except (KeyError, TypeError) as exc:
            raise ModelAssetError(
                f"Model lock {self.path} has no {section} entry for {key!r}. Re-run model preparation."
            ) from exc
        if not isinstance(entry, Mapping):
            raise ModelAssetError(f"Malformed {section}.{key} entry in {self.path}")
        return entry

    def path_for(self, section: str, key: str) -> Path:
        entry = self._entry(section, key)
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ModelAssetError(f"Malformed path for {section}.{key} in {self.path}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.path.parent / path
        path = path.resolve()
        if not path.exists():
            raise ModelAssetError(
                f"Missing prepared model path for {section}.{key}: {path}. Re-run model preparation."
            )
        required = entry.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ModelAssetError(f"Malformed required list for {section}.{key} in {self.path}")
        missing = [path / item for item in required if not (path / item).is_file()]
        if missing:
            raise ModelAssetError(
                f"Incomplete prepared model for {section}.{key}; missing {missing[0]}. Re-run model preparation."
            )
        return path

    def base_model_path(self, key: str) -> Path:
        return self.path_for("base_models", key)

    def checkpoint_path(self, key: str) -> Path:
        return self.path_for("checkpoints", key)

    def metric_path(self, key: str) -> Path:
        return self.path_for("metrics", key)

def required_files(payload: Mapping[str, Any]) -> list[tuple[str, Path, tuple[str, ...]]]:
    requirements: list[tuple[str, Path, tuple[str, ...]]] = []
    for section in ("base_models", "checkpoints", "metrics"):
        entries = payload.get(section, {})
        if not isinstance(entries, Mapping):
            raise ModelAssetError(f"Model lock section {section!r} must be a mapping")
        for key, raw_entry in entries.items():
            if not isinstance(raw_entry, Mapping) or not isinstance(raw_entry.get("path"), str):
                raise ModelAssetError(f"Malformed {section}.{key} entry")
            required = raw_entry.get("required", [])
            if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
                raise ModelAssetError(f"Malformed required list for {section}.{key}")
            requirements.append((f"{section}.{key}", Path(raw_entry["path"]), tuple(required)))
    return requirements


def verify_lock(lock: ModelLock) -> list[str]:
    errors: list[str] = []
    for label, root, required in required_files(lock.payload):
        if not root.is_absolute():
            root = (lock.path.parent / root).resolve()
        if not root.exists():
            errors.append(f"{label}: missing path {root}")
            continue
        for relative_path in required:
            candidate = root / relative_path if root.is_dir() else root
            if not candidate.is_file():
                errors.append(f"{label}: missing required file {candidate}")
    return errors
