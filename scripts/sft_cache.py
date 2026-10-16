"""Reference-SFT latent cache helpers for SD3 training."""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

SFT_CACHE_VERSION = 1


@dataclass
class SftEpochData:
    latents_cpu: torch.Tensor
    prompt_embeds_cpu: torch.Tensor
    pooled_embeds_cpu: torch.Tensor
    endpoint_prompt_indices_cpu: torch.Tensor
    metrics: dict

    def __len__(self):
        return len(self.latents_cpu)


def sft_cache_metadata(prompt, config):
    return {
        "cache_version": SFT_CACHE_VERSION,
        "prompt": prompt,
        "model_name": config.model_name,
        "resolution": config.resolution,
        "num_train_inference_steps": config.num_train_inference_steps,
        "train_guidance_scale": config.train_guidance_scale,
        "sft_num_samples_per_prompt": config.sft_num_samples_per_prompt,
    }


def sft_cache_key(metadata):
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sft_cache_seed(cache_key):
    return int(cache_key[:16], 16) % (2**63 - 1)


def load_sft_cache_file(cache_path, config):
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or "latents" not in payload:
        raise ValueError(f"Invalid SFT cache payload: {cache_path}")
    latents = payload["latents"]
    if latents.shape[0] != config.sft_num_samples_per_prompt:
        raise ValueError(
            f"SFT cache sample count mismatch for {cache_path}: "
            f"expected {config.sft_num_samples_per_prompt}, found {latents.shape[0]}"
        )
    return latents


def acquire_sft_cache_lock(cache_path, timeout_seconds=3600, poll_seconds=1.0):
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    start = time.time()
    while True:
        if cache_path.exists():
            return None
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - start > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for SFT cache lock: {lock_path}")
            time.sleep(poll_seconds)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"pid={os.getpid()} time={time.time()}\n")
        return lock_path


def generate_reference_latents(pipe, prompt, neg_embeds, neg_pooled, config, accelerator, adapter_host_fn, seed):
    count = config.sft_num_samples_per_prompt
    bs = config.sampling_batch_size // 2 if config.train_guidance_scale > 1.0 else config.sampling_batch_size
    bs = min(bs, count)

    with accelerator.autocast():
        prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
            [prompt], prompt_2=None, prompt_3=None, do_classifier_free_guidance=False
        )
    embeds = prompt_embeds.repeat_interleave(count, 0)
    pooled = pooled_embeds.repeat_interleave(count, 0)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    all_latents = []
    host = adapter_host_fn(pipe.transformer)
    with host.disable_adapter():
        for i in range(0, count, bs):
            batch_embeds = embeds[i:i + bs]
            batch_pooled = pooled[i:i + bs]
            if config.train_guidance_scale > 1.0:
                batch_neg_embeds = neg_embeds.expand(batch_embeds.shape[0], -1, -1)
                batch_neg_pooled = neg_pooled.expand(batch_pooled.shape[0], -1)
            else:
                batch_neg_embeds = batch_neg_pooled = None

            with accelerator.autocast():
                terminal_latent = pipe(
                    prompt_embeds=batch_embeds,
                    negative_prompt_embeds=batch_neg_embeds,
                    pooled_prompt_embeds=batch_pooled,
                    negative_pooled_prompt_embeds=batch_neg_pooled,
                    height=config.resolution,
                    width=config.resolution,
                    num_inference_steps=config.num_train_inference_steps,
                    guidance_scale=config.train_guidance_scale,
                    generator=generator,
                    output_type="latent",
                    return_dict=False,
                )[0]
            all_latents.append(terminal_latent.detach().cpu())
    return torch.cat(all_latents)


def get_or_create_sft_latents(pipe, prompt, neg_embeds, neg_pooled, config, accelerator, adapter_host_fn):
    metadata = sft_cache_metadata(prompt, config)
    cache_key = sft_cache_key(metadata)
    cache_dir = Path(config.sft_ref_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.pt"

    if cache_path.exists():
        return load_sft_cache_file(cache_path, config), True

    lock_path = acquire_sft_cache_lock(cache_path)
    if lock_path is None:
        return load_sft_cache_file(cache_path, config), True

    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.{accelerator.process_index}.tmp")
    try:
        if cache_path.exists():
            return load_sft_cache_file(cache_path, config), True
        latents = generate_reference_latents(
            pipe,
            prompt,
            neg_embeds,
            neg_pooled,
            config,
            accelerator,
            adapter_host_fn,
            seed=sft_cache_seed(cache_key),
        )
        torch.save({"metadata": metadata, "latents": latents}, tmp_path)
        os.replace(tmp_path, cache_path)
        return latents, False
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        if lock_path.exists():
            lock_path.unlink()


def sample_sft_epoch(pipe, prompts, neg_embeds, neg_pooled, config, accelerator, adapter_host_fn):
    with accelerator.autocast():
        prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
            prompts, prompt_2=None, prompt_3=None, do_classifier_free_guidance=False
        )

    all_latents = []
    endpoint_prompt_indices = []
    cache_hits = 0
    cache_misses = 0
    for prompt_idx, prompt in enumerate(prompts):
        latents, hit = get_or_create_sft_latents(
            pipe, prompt, neg_embeds, neg_pooled, config, accelerator, adapter_host_fn
        )
        all_latents.append(latents)
        endpoint_prompt_indices.extend([prompt_idx] * len(latents))
        if hit:
            cache_hits += 1
        else:
            cache_misses += 1

    latents_cpu = torch.cat(all_latents).cpu()
    endpoint_prompt_indices_cpu = torch.tensor(endpoint_prompt_indices, dtype=torch.long)
    assert len(latents_cpu) == len(endpoint_prompt_indices_cpu)

    metrics = {"sft_cache_hits": cache_hits, "sft_cache_misses": cache_misses}
    return SftEpochData(
        latents_cpu=latents_cpu,
        prompt_embeds_cpu=prompt_embeds.detach().cpu(),
        pooled_embeds_cpu=pooled_embeds.detach().cpu(),
        endpoint_prompt_indices_cpu=endpoint_prompt_indices_cpu,
        metrics=metrics,
    )
