"""Base-model reference cache for the FD reward: key, layout, loaders, null-FD.

The reference distribution of the Fréchet-distance term is the base model
(LoRA adapters disabled) sampled with exactly the training settings.
``scripts/fd_prepare.py`` generates those samples once, embeds them with every
representation space, computes the per-space null-FD and writes the cache;
training only reads.  The cache is keyed on the sampling settings alone, never
on an encoder::

    <fd_cache_dir>/<key>/
      metadata.json          # cache_version, model_name, resolution, steps, guidance,
                             # prompts, epoch sizes, fd_init_epochs, fd_reference_seed
      images/epoch_NN.pt     # uint8 (epoch_size, 3, H, W), one file per reference epoch
      features/<space>.pt    # float32 (N, d) in the same order, plus {space, dim, N}
      null_fd.json           # version, split spec, D_i per space

Rows are in generation order: rank-major, whole prompt groups contiguous,
epoch by epoch.  ``residual_features`` and the even/odd split rely on that.
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

import fd_encoders
import fd_reward

FD_REFERENCE_CACHE_VERSION = 2
NULL_FD_VERSION = 1
NULL_FD_SPLIT = "even_odd_epochs"
PREPARE_COMMAND = "python scripts/fd_prepare.py <config>"


# ---------------------------------------------------------------------------
# Key and layout
# ---------------------------------------------------------------------------


def fd_reference_seed(config):
    """Seed of the reference sampler: a stream disjoint from training's."""
    if config.fd_reference_seed is not None:
        return int(config.fd_reference_seed)
    payload = f"{config.seed}:fd_reference".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def fd_reference_metadata(config):
    return {
        "cache_version": FD_REFERENCE_CACHE_VERSION,
        "model_name": config.model_name,
        "resolution": config.resolution,
        "num_train_inference_steps": config.num_train_inference_steps,
        "train_guidance_scale": config.train_guidance_scale,
        "prompts": config.prompts,
        "num_prompts_per_epoch": config.num_prompts_per_epoch,
        "num_samples_per_prompt": config.num_samples_per_prompt,
        "fd_init_epochs": config.fd_init_epochs,
        "fd_reference_seed": fd_reference_seed(config),
    }


def fd_reference_key(metadata):
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fd_reference_dir(config):
    return Path(config.fd_cache_dir) / fd_reference_key(fd_reference_metadata(config))


def reference_epoch_size(config):
    return config.num_prompts_per_epoch * config.num_samples_per_prompt


def expected_num_samples(config):
    return config.fd_init_epochs * reference_epoch_size(config)


def metadata_path(directory):
    return Path(directory) / "metadata.json"


def image_path(directory, epoch):
    return Path(directory) / "images" / f"epoch_{int(epoch):02d}.pt"


def feature_path(directory, space):
    return Path(directory) / "features" / f"{space}.pt"


def null_fd_path(directory):
    return Path(directory) / "null_fd.json"


def lock_path(directory):
    directory = Path(directory)
    return directory.with_name(directory.name + ".lock")


# ---------------------------------------------------------------------------
# Atomic writes and the lock
# ---------------------------------------------------------------------------


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        writer(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_save(payload, path):
    _atomic_write(path, lambda tmp: torch.save(payload, tmp))


def atomic_write_json(payload, path):
    _atomic_write(path, lambda tmp: tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"))


def acquire_lock(path, timeout_seconds=6 * 3600, poll_seconds=2.0):
    """Create ``path`` exclusively, waiting for a concurrent holder to finish."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - start > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for the FD reference lock: {path}")
            time.sleep(poll_seconds)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"pid={os.getpid()} time={time.time()}\n")
        return path


def release_lock(path):
    path = Path(path)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Writers (used by fd_prepare.py)
# ---------------------------------------------------------------------------


def write_metadata(directory, metadata):
    atomic_write_json(metadata, metadata_path(directory))


def write_images(directory, epoch, images, cache_key):
    if images.dtype != torch.uint8 or images.ndim != 4:
        raise ValueError(f"reference images must be uint8 (n, 3, H, W), got {images.dtype} {tuple(images.shape)}")
    atomic_save({"epoch": int(epoch), "cache_key": cache_key, "images": images.cpu()}, image_path(directory, epoch))


def write_features(directory, space, features, cache_key):
    features = features.detach().cpu().to(torch.float32)
    payload = {
        "space": space,
        "dim": int(features.shape[1]),
        "N": int(features.shape[0]),
        "cache_key": cache_key,
        "features": features,
    }
    atomic_save(payload, feature_path(directory, space))


def write_null_fd(directory, payload):
    atomic_write_json(payload, null_fd_path(directory))


# ---------------------------------------------------------------------------
# Loaders (used by training)
# ---------------------------------------------------------------------------


def _refusal(config, detail):
    return RuntimeError(
        f"FD reference cache problem: {detail}\n"
        f"Expected directory: {fd_reference_dir(config)}\n"
        f"Prepare it once with: {PREPARE_COMMAND} (multi-GPU: accelerate launch --multi_gpu "
        f"--num_processes N scripts/fd_prepare.py <config>); recompute null_fd.json alone "
        f"with --stats-only."
    )


def check_metadata(config):
    """Return the cache directory after verifying ``metadata.json`` matches ``config``."""
    directory = fd_reference_dir(config)
    path = metadata_path(directory)
    if not path.exists():
        raise _refusal(config, f"{path} is missing")
    metadata = json.loads(path.read_text())
    expected = fd_reference_metadata(config)
    if fd_reference_key(metadata) != fd_reference_key(expected):
        raise _refusal(config, f"{path} does not match the config")
    return directory


def load_reference_features(config, spaces, *, map_location="cpu"):
    """Load and validate reference features for every requested space.

    ``map_location="meta"`` performs the same payload and shape validation
    without allocating the feature tensors.  The verification CLI uses that
    mode while training keeps the default float32 CPU tensors.
    """
    directory = check_metadata(config)
    key = directory.name
    result = {}
    for space in spaces:
        dim = fd_encoders.resolve_fd_type(space).dim
        path = feature_path(directory, space)
        if not path.is_file() or not path.stat().st_size:
            raise _refusal(config, f"feature file {path} is missing")
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(payload, dict) or "features" not in payload:
            raise _refusal(config, f"{path} has an invalid payload")
        features = payload["features"]
        expected = (expected_num_samples(config), dim)
        if tuple(features.shape) != expected:
            raise _refusal(config, f"{path} has shape {tuple(features.shape)}, expected {expected}")
        if payload.get("space") != space or payload.get("cache_key") != key:
            raise _refusal(config, f"{path} belongs to space {payload.get('space')!r}, key {payload.get('cache_key')!r}")
        result[space] = features.to(torch.float32)
    return result


def load_null_fd(config, spaces):
    """``{space: D}`` from ``null_fd.json``, checked against the config's numerics."""
    directory = check_metadata(config)
    path = null_fd_path(directory)
    if not path.exists():
        raise _refusal(config, f"{path} is missing")
    payload = json.loads(path.read_text())
    if payload.get("version") != NULL_FD_VERSION or payload.get("split") != NULL_FD_SPLIT:
        raise _refusal(config, f"{path} has version {payload.get('version')!r} / split {payload.get('split')!r}, "
                               f"expected {NULL_FD_VERSION} / {NULL_FD_SPLIT!r}")
    for key, value in (
        ("fd_init_epochs", config.fd_init_epochs),
        ("shrinkage", float(config.fd_shrinkage)),
        ("eig_floor", float(config.fd_eig_floor)),
    ):
        if payload.get(key) != value:
            raise _refusal(config, f"{path} was computed with {key}={payload.get(key)!r}, the config has {value!r}")
    result = {}
    for space in spaces:
        entry = payload.get("spaces", {}).get(space)
        if entry is None:
            raise _refusal(config, f"{path} has no entry for space {space!r}")
        value = float(entry["D"])
        if not math.isfinite(value) or value <= 0.0:
            raise _refusal(config, f"{path} has D={value!r} for {space!r}")
        result[space] = value
    return result


# ---------------------------------------------------------------------------
# Null-FD
# ---------------------------------------------------------------------------


def even_odd_split(features, epoch_size):
    """Split rows by reference epoch into the even and the odd epochs.

    Both halves keep whole prompt groups and the generation order.
    """
    n, d = features.shape
    epoch_size = int(epoch_size)
    if epoch_size < 1 or n % epoch_size != 0:
        raise ValueError(f"number of rows ({n}) must be a positive multiple of the epoch size ({epoch_size})")
    epochs = n // epoch_size
    if epochs < 2:
        raise ValueError(f"the null-FD needs at least two reference epochs, got {epochs}")
    by_epoch = features.view(epochs, epoch_size, d)
    return by_epoch[0::2].reshape(-1, d), by_epoch[1::2].reshape(-1, d)


def compute_null_fd(features, epoch_size, group_size, config):
    """``D``: the FD between the two halves at the config's shrinkage and floor.

    The value a policy identical to the base would score, nonzero only
    through finite-sample noise.  Returns a dict with ``D``, its mean and
    Bures parts, and the half sizes.
    """
    half_a, half_b = even_odd_split(features.to(torch.float64), epoch_size)
    reference = fd_reward.FdReference.from_features(half_a, group_size, config)
    moments = fd_reward.FdMoments()
    moments.warm_start(half_b, group_size)
    out = fd_reward.compute_fd_reward(half_b[: int(group_size)], group_size, moments, reference, config)
    return {
        "D": float(out.fd),
        "fd_mean": float(out.fd_mean),
        "fd_cov": float(out.fd_cov),
        "n_a": int(half_a.shape[0]),
        "n_b": int(half_b.shape[0]),
    }


def null_fd_payload(config, cache_key, per_space):
    return {
        "version": NULL_FD_VERSION,
        "split": NULL_FD_SPLIT,
        "cache_key": cache_key,
        "fd_init_epochs": config.fd_init_epochs,
        "shrinkage": float(config.fd_shrinkage),
        "eig_floor": float(config.fd_eig_floor),
        "spaces": per_space,
    }


__all__ = [
    "FD_REFERENCE_CACHE_VERSION", "NULL_FD_SPLIT", "NULL_FD_VERSION", "PREPARE_COMMAND", "acquire_lock",
    "check_metadata", "compute_null_fd", "even_odd_split", "expected_num_samples", "fd_reference_dir",
    "fd_reference_key", "fd_reference_metadata", "fd_reference_seed", "feature_path", "image_path",
    "load_null_fd", "load_reference_features", "lock_path", "metadata_path", "null_fd_path",
    "null_fd_payload", "reference_epoch_size", "release_lock", "write_features", "write_images",
    "write_metadata", "write_null_fd",
]
