"""Small helpers for reducing scalar metrics across Accelerate processes."""

import math
from collections.abc import Mapping

import torch


def _as_scalar_tensor(accelerator, value):
    if torch.is_tensor(value):
        value = value.detach().to(device=accelerator.device, dtype=torch.float64)
        if value.numel() != 1:
            raise ValueError(f"Expected a scalar metric, got shape {tuple(value.shape)}")
        return value.reshape(())
    return torch.tensor(float(value), device=accelerator.device, dtype=torch.float64)


def reduce_scalar_dict(accelerator, metrics: Mapping[str, object], reduction="mean"):
    """Reduce a same-keyed dictionary of scalar metrics on every process."""
    if not metrics:
        return {}
    keys = tuple(sorted(metrics))
    values = torch.stack([_as_scalar_tensor(accelerator, metrics[key]) for key in keys])
    reduced = accelerator.reduce(values, reduction=reduction)
    return {key: value.item() for key, value in zip(keys, reduced)}


def reduce_tensor_means(accelerator, tensors: Mapping[str, torch.Tensor]):
    """Return sample-weighted global means for possibly uneven local tensors."""
    if not tensors:
        return {}
    keys = tuple(sorted(tensors))
    stats = []
    for key in keys:
        value = tensors[key].detach()
        stats.append(
            torch.stack(
                (
                    value.to(device=accelerator.device, dtype=torch.float64).sum(),
                    torch.tensor(value.numel(), device=accelerator.device, dtype=torch.float64),
                )
            )
        )
    stats = accelerator.reduce(torch.stack(stats), reduction="sum")
    if torch.any(stats[:, 1] == 0):
        empty_keys = [key for key, stat in zip(keys, stats) if stat[1].item() == 0]
        raise ValueError(f"Cannot average empty distributed metrics: {empty_keys}")
    return {key: (stat[0] / stat[1]).item() for key, stat in zip(keys, stats)}


def reduce_masked_means(accelerator, entries: Mapping[str, tuple]):
    """Global mean of ``values[mask]`` per key across processes.

    ``entries`` maps a key to ``(values, mask)`` of equal shape.  Unlike
    ``reduce_tensor_means`` an empty subset is allowed: a key whose mask is
    empty on every process comes back as NaN rather than raising, so buckets
    that are legitimately empty on some epochs (for example "solved prompts"
    early in training) leave a gap in the log instead of aborting the run.
    """
    if not entries:
        return {}
    keys = tuple(sorted(entries))
    stats = []
    for key in keys:
        values, mask = entries[key]
        values = values.detach().to(device=accelerator.device, dtype=torch.float64)
        mask = mask.detach().to(device=accelerator.device, dtype=torch.bool)
        if values.shape != mask.shape:
            raise ValueError(f"{key}: values {tuple(values.shape)} and mask {tuple(mask.shape)} differ in shape")
        stats.append(torch.stack((values[mask].sum(), mask.sum().to(torch.float64))))
    stats = accelerator.reduce(torch.stack(stats), reduction="sum")
    reduced = {}
    for key, stat in zip(keys, stats):
        count = stat[1].item()
        reduced[key] = (stat[0] / stat[1]).item() if count > 0 else math.nan
    return reduced


def reduce_max(accelerator, value):
    """Return the maximum scalar value observed on any process."""
    local = _as_scalar_tensor(accelerator, value).reshape(1)
    return accelerator.gather(local).max().item()
