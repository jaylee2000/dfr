"""One-shot preparation of the FD reference cache.

Generates the base-model reference images once, embeds them with every
representation space, computes the per-space null-FD and writes the cache
that every training run reads (layout in ``fd_reference.py``)::

    python scripts/fd_prepare.py <config> [--stats-only] [--spaces a,b] [--key value ...]
    accelerate launch --multi_gpu --num_processes N scripts/fd_prepare.py <config> [...]

Config resolution is the training one, so the cache key is computed from
exactly the values training will compute.  Three stages, each skipped when
its output exists:

- **generate**: for each missing reference epoch, the training batch sampler
  on the reference seed, adapters disabled, ``generate_images`` at the
  training settings, the images all-gathered and written by the main process
  as one uint8 file.  A killed job re-runs only the missing epochs.
- **embed**: one space at a time, sharded across ranks by epoch file; only
  spaces without a feature file run (``--spaces`` restricts the set).
- **stats**: main process only; the even/odd null-FD of every feature file
  present, written to ``null_fd.json``.  ``--stats-only`` runs this alone.

The lock covers the whole run: a second concurrent launch waits, then finds
the outputs and skips every stage.
"""

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

import fd_encoders
import fd_reference
import fd_reward
from batch_sampler import PromptSeedBatchSampler
from config_loader import load_config, validate_training_options
from generation import generate_images
from loss_utils import adapter_host


def resolve_config(config, num_processes):
    # Imported here: training_sd3 imports fd_reference at load time.
    import training_sd3

    validate_training_options(config)
    config = training_sd3.apply_reproducibility_config_defaults(config)
    config = fd_reward.apply_fd_config_defaults(config)
    training_sd3.derive_local_sizes(config, num_processes)
    return config


def missing_epochs(directory, config):
    return [e for e in range(config.fd_init_epochs) if not fd_reference.image_path(directory, e).exists()]


# ---------------------------------------------------------------------------
# Stage 1: generate
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_stage(config, accelerator, directory, cache_key, epochs):
    import training_sd3

    pipe = training_sd3.build_pipeline(config, accelerator)
    pipe.to(accelerator.device)
    train_prompts = training_sd3.load_prompts(Path("prompts") / config.prompts / "train.txt")
    with accelerator.autocast():
        neg_embeds, _, neg_pooled, _ = pipe.encode_prompt(
            [""], prompt_2=None, prompt_3=None, do_classifier_free_guidance=False
        )
    num_samples = config.num_samples_per_prompt
    sampler = PromptSeedBatchSampler(
        dataset_size=len(train_prompts),
        num_prompts_per_epoch=config.num_prompts_per_epoch,
        num_samples_per_prompt=num_samples,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=fd_reference.fd_reference_seed(config),
    )
    epoch_size = fd_reference.reference_epoch_size(config)
    host = adapter_host(pipe.transformer)
    t_start = time.time()
    with host.disable_adapter():
        for index, epoch in enumerate(epochs):
            sampler.set_epoch(epoch)
            batch = next(iter(sampler))
            prompts = [train_prompts[prompt_index] for prompt_index, _ in batch]
            seeds = [seed for _, seeds_for_prompt in batch for seed in seeds_for_prompt]
            with accelerator.autocast():
                prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
                    prompts, prompt_2=None, prompt_3=None, do_classifier_free_guidance=False
                )
            embeds = prompt_embeds.repeat_interleave(num_samples, 0)
            pooled = pooled_embeds.repeat_interleave(num_samples, 0)
            _, images = generate_images(
                pipe, embeds, pooled, seeds, neg_embeds, neg_pooled, config, accelerator
            )
            images_u8 = (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            gathered = accelerator.gather(images_u8).cpu()
            # Rows must be rank-major with whole prompt groups contiguous: the
            # same order sample_epoch's features have, which residual_features
            # and the even/odd split rely on.
            n_local = images_u8.shape[0]
            local = slice(accelerator.process_index * n_local, (accelerator.process_index + 1) * n_local)
            assert torch.equal(gathered[local], images_u8.cpu()), "gathered images are not rank-major"
            if accelerator.is_main_process:
                assert gathered.shape[0] == epoch_size, (gathered.shape, epoch_size)
                fd_reference.write_images(directory, epoch, gathered, cache_key)
            del images, images_u8, gathered
            accelerator.print(
                f"[FD prepare] generated epoch {epoch:02d} ({index + 1}/{len(epochs)}, "
                f"{time.time() - t_start:.0f}s)"
            )
    accelerator.wait_for_everyone()
    del pipe
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 2: embed
# ---------------------------------------------------------------------------


def _part_path(directory, space, epoch):
    return Path(directory) / "features" / f".{space}.epoch_{int(epoch):02d}.part"


@torch.no_grad()
def embed_epoch_file(encoder, path, batch_size, device):
    payload = torch.load(path, map_location="cpu")
    images = payload["images"]
    chunks = []
    for start in range(0, images.shape[0], batch_size):
        chunk = images[start:start + batch_size].to(device).float() / 255.0
        chunks.append(encoder.embed(chunk).cpu())
    return torch.cat(chunks)


@torch.no_grad()
def embed_stage(config, accelerator, directory, cache_key, spaces):
    epochs = list(range(config.fd_init_epochs))
    world, rank = accelerator.num_processes, accelerator.process_index
    for space in spaces:
        t_start = time.time()
        encoder = fd_encoders.FdEncoder(space, config.fd_encoder_cache_dir, config.fd_batch_size, accelerator.device)
        mine = [e for e in epochs if e % world == rank]
        for epoch in mine:
            features = embed_epoch_file(
                encoder, fd_reference.image_path(directory, epoch), config.fd_batch_size, accelerator.device
            )
            fd_reference.atomic_save(features, _part_path(directory, space, epoch))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            parts = [torch.load(_part_path(directory, space, e), map_location="cpu") for e in epochs]
            features = torch.cat(parts)
            assert features.shape == (fd_reference.expected_num_samples(config), encoder.dim), features.shape
            fd_reference.write_features(directory, space, features, cache_key)
            for e in epochs:
                _part_path(directory, space, e).unlink()
            norm = features.norm(dim=1).mean().item()
        accelerator.wait_for_everyone()
        del encoder
        torch.cuda.empty_cache()
        accelerator.print(
            f"[FD prepare] embedded {space} ({time.time() - t_start:.0f}s)"
            + (f", mean feature norm {norm:.4g}" if accelerator.is_main_process else "")
        )


# ---------------------------------------------------------------------------
# Stage 3: stats
# ---------------------------------------------------------------------------


def stats_stage(config, directory, cache_key):
    """Null-FD of every feature file present; returns the JSON payload."""
    present = [s for s in fd_encoders.FD_ENCODERS if fd_reference.feature_path(directory, s).exists()]
    if not present:
        raise RuntimeError(f"No feature files under {directory / 'features'}; run the embed stage first")
    features = fd_reference.load_reference_features(config, present)
    epoch_size = fd_reference.reference_epoch_size(config)
    group_size = config.num_samples_per_prompt
    per_space = {}
    for space in present:
        t_start = time.time()
        entry = fd_reference.compute_null_fd(features[space], epoch_size, group_size, config)
        entry["feature_norm_mean"] = features[space].norm(dim=1).mean().item()
        per_space[space] = entry
        print(f"[FD prepare] null-FD {space}: D={entry['D']:.6g} ({time.time() - t_start:.0f}s)", flush=True)
    payload = fd_reference.null_fd_payload(config, cache_key, per_space)
    fd_reference.write_null_fd(directory, payload)
    return payload


def print_table(payload):
    print("space              D (null FD)   fd_mean      fd_cov       |f| mean   n_a / n_b")
    for space, entry in payload["spaces"].items():
        print(
            f"{space:18s} {entry['D']:12.5g} {entry['fd_mean']:12.5g} {entry['fd_cov']:12.5g} "
            f"{entry['feature_norm_mean']:10.4g}   {entry['n_a']} / {entry['n_b']}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config, stats_only=False, spaces=None):
    num_processes = int(os.environ.get("WORLD_SIZE", "1"))
    config = resolve_config(config, num_processes)
    if config.fd_init_epochs < 2:
        raise ValueError("fd_init_epochs must be at least 2: the null-FD splits the reference epochs in two halves")
    spaces = list(spaces or fd_encoders.FD_ENCODERS)
    for space in spaces:
        fd_encoders.resolve_fd_type(space)

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=6))],
    )
    metadata = fd_reference.fd_reference_metadata(config)
    cache_key = fd_reference.fd_reference_key(metadata)
    directory = fd_reference.fd_reference_dir(config)
    lock = None
    if accelerator.is_main_process:
        lock = fd_reference.acquire_lock(fd_reference.lock_path(directory))
        directory.mkdir(parents=True, exist_ok=True)
        if not fd_reference.metadata_path(directory).exists():
            fd_reference.write_metadata(directory, metadata)
    accelerator.wait_for_everyone()
    fd_reference.check_metadata(config)
    accelerator.print(f"[FD prepare] cache {directory} (key {cache_key[:12]}, {num_processes} processes)")

    try:
        if not stats_only:
            epochs = missing_epochs(directory, config)
            if epochs:
                accelerator.print(
                    f"[FD prepare] generating {len(epochs)}/{config.fd_init_epochs} reference epochs of "
                    f"{fd_reference.reference_epoch_size(config)} images"
                )
                generate_stage(config, accelerator, directory, cache_key, epochs)
            else:
                accelerator.print("[FD prepare] all reference image files present; skipping generation")
            accelerator.wait_for_everyone()
            to_embed = [s for s in spaces if not fd_reference.feature_path(directory, s).exists()]
            if to_embed:
                accelerator.print(f"[FD prepare] embedding {to_embed}")
                embed_stage(config, accelerator, directory, cache_key, to_embed)
            else:
                accelerator.print("[FD prepare] all requested feature files present; skipping embedding")
            accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            payload = stats_stage(config, directory, cache_key)
            print_table(payload)
            print(f"[FD prepare] wrote {fd_reference.null_fd_path(directory)}", flush=True)
        accelerator.wait_for_everyone()
    finally:
        if lock is not None:
            fd_reference.release_lock(lock)


if __name__ == "__main__":
    import training_sd3

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="config name (without .yaml)")
    parser.add_argument("--stats-only", action="store_true", help="recompute null_fd.json from the feature files")
    parser.add_argument("--spaces", type=str, default=None, help="comma-separated registry names; default all six")
    args, overrides = parser.parse_known_args()
    config = load_config(args.config)
    config.update(training_sd3.parse_overrides(overrides))
    spaces = [s for s in args.spaces.split(",") if s] if args.spaces else None
    main(argparse.Namespace(**config), stats_only=args.stats_only, spaces=spaces)
