from __future__ import annotations

import os
from pathlib import Path


def _required_absolute_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set in local.env")
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"{name} must be an absolute path, got: {value}")
    return path


RAM_ROOT = _required_absolute_path("RAM_ROOT")
EVAL_ROOT = _required_absolute_path("EVAL_ROOT")
OUTPUTS_ROOT = _required_absolute_path("OUTPUTS_ROOT")

# ROOT remains available for modules that resolve repository-owned files.
ROOT = EVAL_ROOT

DATASETS = ("geneval", "drawbench", "ocr")
PER_PROMPT_DATASETS = DATASETS

DEFAULT_MANIFEST_DIR = OUTPUTS_ROOT / "manifests"
DEFAULT_IMAGE_ROOT = OUTPUTS_ROOT / "images"
DEFAULT_METRIC_ROOT = OUTPUTS_ROOT / "metrics"

# Model preparation is removed in a later refactor. Until then, keep its
# storage deterministic without exposing additional local.env overrides.
DEFAULT_MODEL_ROOT = EVAL_ROOT / "models"
DEFAULT_MODEL_LOCK = DEFAULT_MODEL_ROOT / "models.lock.json"

DEFAULT_PROMPT_SOURCES = {
    "geneval": EVAL_ROOT / "prompts" / "geneval" / "test_metadata.jsonl",
    "drawbench": EVAL_ROOT / "prompts" / "drawbench" / "test.txt",
    "ocr": RAM_ROOT / "prompts" / "ocr" / "test.txt",
}

# Backward-compatible Python name; these paths are no longer configurable.
BUNDLED_PROMPT_SOURCES = DEFAULT_PROMPT_SOURCES

LORA_TARGET_MODULES = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
]
