"""Reinforce Adjoint Matching (RAM) training script for Stable Diffusion 3.5 Medium.

This is the main entry point for the text-to-image experiments in the paper.
Launch with ``accelerate launch``::

    accelerate launch scripts/training_sd3.py geneval/ram

The first positional argument names a config under ``configs/`` without the
``.yaml`` suffix (``geneval/ram`` is ``configs/geneval/ram.yaml``). Outputs
(checkpoints + a copy of the resolved config) go to ``outputs/<config_name>``
unless ``--output_dir`` is given. Re-running the same command resumes from the
``latest`` checkpoint inside that directory.
"""

import argparse
import gc
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, broadcast
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model

import reward_models as rewards_module
from batch_sampler import PromptSeedBatchSampler
from config_loader import load_config
from distributed_metrics import (
    reduce_masked_means,
    reduce_max,
    reduce_scalar_dict,
    reduce_tensor_means,
)
import dreamsim_diversity
import energy_scoring
import fd_encoders
import fd_reference
import fd_reward
from generation import generate_images, sampling_batch_size
from grad_norms import (
    GradNormMonitor,
    mse_vjp_surrogates,
    ram_residual_terms,
    should_monitor_epoch,
    validate_monitor_interval,
)
from loss_utils import adapter_host, model_forward
import sft_cache
import sft_weights
from vendi_diversity import grouped_vendi_scores

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RAM maintains three LoRA adapters on top of SD3's transformer:
#   default     — the trainable policy
#   old         — a lagged copy used as the "old" target in the RAM loss
#   evaluation  — a more lagged EMA copy used at test time (the policy the
#                 paper's reported numbers are computed with)
LORA_TARGET_MODULES = [
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
]
DTYPE_MAP = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def release_cuda_memory():
    """Collect unreachable modules and tensors before returning to training.

    Validation creates heavyweight, short-lived scorers (notably
    ImageReward) and complete image batches. Some third-party modules contain
    Python reference cycles, so function return and ``empty_cache`` alone do
    not guarantee that their CUDA tensors have been destroyed.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def apply_main_loss_config(config):
    K = config.num_integral_kl_timesteps
    if isinstance(K, bool) or not isinstance(K, int) or K < 0:
        raise ValueError(f"num_integral_kl_timesteps must be an integer >= 0, got {K!r}")
    if config.integral_kl_timestep_distribution not in ("uniform", "linear_low_bias"):
        raise ValueError(
            "integral_kl_timestep_distribution must be 'uniform' or 'linear_low_bias', "
            f"got {config.integral_kl_timestep_distribution!r}"
        )
    for name in ("integral_kl_weight", "reverse_kl_weight"):
        value = float(getattr(config, name))
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        setattr(config, name, value)
    return config


def apply_wandb_config_defaults(config):
    if not hasattr(config, "wandb_run_name"):
        config.wandb_run_name = None
    return config


def apply_grad_norm_monitor_config_defaults(config):
    if not hasattr(config, "grad_norm_monitor_every_n_epochs"):
        config.grad_norm_monitor_every_n_epochs = 0
    config.grad_norm_monitor_every_n_epochs = validate_monitor_interval(
        config.grad_norm_monitor_every_n_epochs
    )
    return config


def apply_reproducibility_config_defaults(config):
    if not hasattr(config, "seed"):
        config.seed = 0
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {config.seed!r}")
    return config


def apply_validation_config_defaults(config):
    raw_fraction = getattr(config, "eval_prompt_fraction", 1.0)
    if isinstance(raw_fraction, bool):
        raise ValueError(
            f"eval_prompt_fraction must be a finite number in (0, 1], got {raw_fraction!r}"
        )
    try:
        fraction = float(raw_fraction)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"eval_prompt_fraction must be a finite number in (0, 1], got {raw_fraction!r}"
        ) from error
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError(
            f"eval_prompt_fraction must be a finite number in (0, 1], got {raw_fraction!r}"
        )
    config.eval_prompt_fraction = fraction
    return config


def resolve_eval_guidance_scales(config):
    """Resolve and validate the CFG values used by distributed validation."""
    raw_scales = getattr(config, "eval_guidance_scales", None)
    if raw_scales is None:
        raw_scales = [config.eval_guidance_scale]
    if not isinstance(raw_scales, (list, tuple)) or not raw_scales:
        raise ValueError("eval_guidance_scales must be a non-empty list")

    scales = []
    for scale in raw_scales:
        if isinstance(scale, bool):
            raise ValueError(f"Validation guidance scales must be finite numbers, got {scale!r}")
        scale = float(scale)
        if not math.isfinite(scale):
            raise ValueError(f"Validation guidance scales must be finite numbers, got {scale!r}")
        scales.append(scale)

    if len(set(scales)) != len(scales):
        raise ValueError(f"Validation guidance scales must be unique, got {scales}")
    config.eval_guidance_scales = scales
    return scales


def resolve_wandb_run_name(config, output_dir):
    return config.wandb_run_name or output_dir.name


def apply_sft_config_defaults(config):
    defaults = {
        "sft_ref_cache_dir": None,
        "sft_loss_weight": 0.0,
        "sft_loss_weight_fn": None,
        "sft_num_samples_per_prompt": None,
        "sft_num_loss_targets_per_sample": None,
    }
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)

    if config.sft_loss_weight is None:
        config.sft_loss_weight = 0.0
    config.sft_loss_weight = float(config.sft_loss_weight)

    if config.sft_num_samples_per_prompt is None:
        config.sft_num_samples_per_prompt = config.num_samples_per_prompt
    if config.sft_num_loss_targets_per_sample is None:
        config.sft_num_loss_targets_per_sample = config.num_loss_targets_per_sample
    config.sft_num_samples_per_prompt = int(config.sft_num_samples_per_prompt)
    config.sft_num_loss_targets_per_sample = int(config.sft_num_loss_targets_per_sample)

    if config.sft_num_samples_per_prompt <= 0:
        raise ValueError("sft_num_samples_per_prompt must be positive")
    if config.sft_num_loss_targets_per_sample <= 0:
        raise ValueError("sft_num_loss_targets_per_sample must be positive")
    if sft_is_enabled(config) and not config.sft_ref_cache_dir:
        raise ValueError("sft_ref_cache_dir is required when SFT regularization is enabled")
    return config


def sft_is_enabled(config):
    return bool(config.sft_loss_weight_fn) or float(config.sft_loss_weight) != 0.0


def resolve_sft_loss_weight(config, epoch):
    if not config.sft_loss_weight_fn:
        return float(config.sft_loss_weight)
    return sft_weights.resolve(config.sft_loss_weight_fn, epoch, float(config.sft_loss_weight), config)


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


def set_adapter(model, name):
    adapter_host(model).set_adapter(name)


def collect_adapter_params(transformer):
    """Return three lists of trainable parameters, one per adapter."""
    params = {}
    for name in ("default", "old", "evaluation"):
        set_adapter(transformer, name)
        params[name] = [p for p in transformer.parameters() if p.requires_grad]
    set_adapter(transformer, "default")
    return params["default"], params["old"], params["evaluation"]


def sync_lagged_params(source, target):
    for s, t in zip(source, target, strict=True):
        t.data.copy_(s.detach().data)


def update_lagged_params(source, target, step, decay, warmup_rate):
    """EMA update with linear warmup: ``target ← mix·target + (1−mix)·source``."""
    if decay is None:
        mix = 0.0
    elif warmup_rate is None:
        mix = decay
    else:
        mix = min(warmup_rate * step, decay)
    for s, t in zip(source, target, strict=True):
        t.data.mul_(mix).add_(s.detach().data, alpha=1.0 - mix)


# ---------------------------------------------------------------------------
# RAM loss
# ---------------------------------------------------------------------------


def sample_timesteps(scheduler, config, num_samples, num_targets_per_sample, device):
    """Sample loss timesteps according to ``config.timestep_sampling``.

    Modes:
        ``weighted`` — sample with replacement from 1000 timesteps with
            linearly decreasing weight in ``t``.
        ``uniform``  — sample without replacement from the high-noise tail
            ``[min_timesteps:]`` of the 1000-step grid.
        ``train`` / ``eval`` — sample without replacement from the train/eval
            inference grid.
    """
    if config.timestep_sampling == "train":
        scheduler.set_timesteps(config.num_train_inference_steps, device=device)
    elif config.timestep_sampling == "eval":
        scheduler.set_timesteps(config.num_eval_inference_steps, device=device)
    else:
        scheduler.set_timesteps(1000, device=device)
    timesteps = scheduler.timesteps

    if config.timestep_sampling == "uniform":
        n = len(timesteps) - config.min_timesteps
    elif config.timestep_sampling in ("weighted", "train", "eval"):
        n = len(timesteps)
    else:
        raise ValueError(f"Invalid timestep sampling mode: {config.timestep_sampling}")

    total = num_samples * num_targets_per_sample
    if config.timestep_sampling == "weighted":
        # 1000 timesteps make duplicate draws negligible; sampling with
        # replacement is the simplest correct option.
        weights = 1.0 - torch.arange(n, device=device, dtype=torch.float32) / n
        indices = torch.multinomial(weights / weights.sum(), total, replacement=True)
    else:
        # Without replacement, per sample.
        indices = torch.rand(num_samples, n, device=device).argsort(dim=1)[:, :num_targets_per_sample].reshape(-1)
    return timesteps[indices]


def get_sample_kl(timestep, timestep_dmd, sample):
    sigma = (timestep / 1000).view(-1, 1, 1, 1)
    sigma_dmd = (timestep_dmd / 1000).view(-1, 1, 1, 1)

    same_step = torch.isclose(
        timestep_dmd.to(dtype=torch.float32),
        timestep.to(dtype=torch.float32),
        rtol=0.0,
        atol=1e-6,
    )
    non_same = ~same_step
    sample_kl = sample.clone()
    if non_same.any().item():
        time_ = sigma[non_same]
        time_kl_ = sigma_dmd[non_same]
        sample_ = sample[non_same]
        noise_kl = torch.randn_like(sample_)
        mu_kl = (1 - time_kl_) / (1 - time_)
        sample_kl[non_same] = mu_kl * sample_ + torch.sqrt(
            (1 - mu_kl) * (1 - mu_kl + 2 * mu_kl * time_)
        ) * noise_kl
    return sample_kl


def sample_integral_kl_timesteps(timestep, num_integral_kl_timesteps, distribution_type):
    """Return K independently sampled integral-KL timesteps with shape [K, batch]."""
    if num_integral_kl_timesteps < 0:
        raise ValueError("num_integral_kl_timesteps must be non-negative")
    if distribution_type not in ("uniform", "linear_low_bias"):
        raise ValueError(f"Unknown integral-KL timestep distribution: {distribution_type!r}")

    dmd_low = timestep.to(dtype=torch.float32).ceil()
    timesteps = []
    for _ in range(num_integral_kl_timesteps):
        if distribution_type == "uniform":
            # Uniform integer draw from [timestep, 1000], inclusive.
            timestep_dmd = dmd_low + torch.floor(
                torch.rand_like(dmd_low) * (1001.0 - dmd_low).clamp_min(1.0)
            )
        else:
            # Linearly decreasing density on [timestep, 1000], biased toward timestep.
            dmd_span = (1000.0 - dmd_low).clamp_min(0.0)
            timestep_dmd = dmd_low + torch.floor(
                dmd_span * (1.0 - torch.sqrt(1.0 - torch.rand_like(dmd_low)))
            )
        timesteps.append(timestep_dmd.to(dtype=timestep.dtype))

    if not timesteps:
        return timestep.new_empty((0, *timestep.shape))
    return torch.stack(timesteps)


def get_target_velocity(
    model,
    sample,
    timestep,
    velocity,
    noise,
    latent,
    prompt_embeds,
    pooled_embeds,
    advantages,
    config,
    accelerator,
):
    """Construct the RAM target from reward, integral-KL, and reverse-KL anchors."""
    host = adapter_host(model)
    K = config.num_integral_kl_timesteps
    integral_kl_anchor = torch.zeros_like(velocity)
    output_delta_norm = None

    with torch.no_grad():
        host.set_adapter("old")
        old_v = model_forward(model, sample, timestep, prompt_embeds, pooled_embeds, accelerator)

        # The base-policy forward now also runs when the reverse-KL anchor is off,
        # so that output_delta_norm (the drift away from the base policy) is finite
        # in every run.  With K > 0 the integral-KL block already measures that
        # drift and base_v is never read, so the pass is skipped there.
        base_v = None
        if config.reverse_kl_weight != 0.0 or K == 0:
            host.set_adapter("default")
            with host.disable_adapter():
                base_v = model_forward(model, sample, timestep, prompt_embeds, pooled_embeds, accelerator)

        if K > 0:
            integral_kl_timesteps = sample_integral_kl_timesteps(
                timestep, K, config.integral_kl_timestep_distribution
            )
            sample_kl = torch.cat([get_sample_kl(timestep, t_dmd, sample) for t_dmd in integral_kl_timesteps])
            timestep_kl = integral_kl_timesteps.reshape(-1)
            prompt_embeds_kl = torch.cat([prompt_embeds] * K)
            pooled_embeds_kl = torch.cat([pooled_embeds] * K)

            host.set_adapter("default")
            with host.disable_adapter():
                base_v_kl = model_forward(
                    model, sample_kl, timestep_kl, prompt_embeds_kl, pooled_embeds_kl, accelerator
                ).chunk(K)
            host.set_adapter("default")
            velocity_kl = model_forward(
                model, sample_kl, timestep_kl, prompt_embeds_kl, pooled_embeds_kl, accelerator
            ).chunk(K)
            kl_deltas = [base - current for base, current in zip(base_v_kl, velocity_kl, strict=True)]
            mean_kl_delta = torch.stack(kl_deltas).mean(dim=0)
            integral_kl_anchor = config.integral_kl_weight * mean_kl_delta
            output_delta_norm = torch.stack([delta.square().mean() for delta in kl_deltas]).mean().item()

        reverse_kl_anchor = velocity.detach()
        if config.reverse_kl_weight != 0.0:
            reverse_kl_anchor = reverse_kl_anchor + config.reverse_kl_weight * (base_v - velocity.detach())

        scaled_adv = config.reward_multiplier * advantages.view(-1, 1, 1, 1)
        reward_update = scaled_adv * ((noise - latent) - old_v)
        target_velocity = reward_update + integral_kl_anchor + reverse_kl_anchor

        if output_delta_norm is None:
            output_delta_norm = (velocity.detach() - base_v).square().mean().item()
        # Gap to the sampling policy: both tensors already exist, and the
        # difference is a transient inside no_grad.
        old_policy_delta_norm = (velocity.detach() - old_v).square().mean().item()

    host.set_adapter("default")
    metrics = {
        "output_delta_norm": output_delta_norm,
        "old_policy_delta_norm": old_policy_delta_norm,
        "integral_kl_anchor_norm": integral_kl_anchor.square().mean().item(),
        "reverse_kl_anchor_norm": (velocity.detach() - reverse_kl_anchor).square().mean().item(),
        "num_integral_kl_timesteps": K,
        "integral_kl_weight": config.integral_kl_weight,
        "reverse_kl_weight": config.reverse_kl_weight,
    }
    residual_terms = ram_residual_terms(
        velocity,
        reward_update,
        reverse_kl_anchor,
        integral_kl_anchor if K > 0 else None,
    )
    return target_velocity, metrics, residual_terms


def compute_main_loss(
    model,
    latent,
    prompt_embeds,
    pooled_embeds,
    advantages,
    timestep,
    config,
    accelerator,
    monitor_grad_norms=False,
):
    host = adapter_host(model)
    host.set_adapter("default")

    sigma = (timestep / 1000).view(-1, 1, 1, 1)
    noise = torch.randn_like(latent)
    sample = (1 - sigma) * latent + sigma * noise  # x_t ~ q(x_t | x_0)
    velocity = model_forward(model, sample, timestep, prompt_embeds, pooled_embeds, accelerator)
    target_velocity, target_metrics, residual_terms = get_target_velocity(
        model, sample, timestep, velocity, noise, latent, prompt_embeds, pooled_embeds, advantages, config, accelerator
    )

    per_sample = ((velocity - target_velocity.detach()) ** 2).mean(dim=tuple(range(1, velocity.ndim)))

    K = config.num_loss_targets_per_sample
    if K > 1:
        assert per_sample.shape[0] % K == 0, "loss batch size must be divisible by num_loss_targets_per_sample"
        per_sample = per_sample.view(-1, K).mean(dim=1)
    loss = per_sample.mean()

    with torch.no_grad():
        # Both are means over a batch of latents; the squares/products are
        # transients inside no_grad, and nothing is kept alive.
        velocity_detached = velocity.detach()
        velocity_norm = (velocity_detached**2).mean().item()
        velocity_target_dot = (velocity_detached * target_velocity.detach()).mean().item()

    metrics = {
        "loss": loss.item(),
        "target_velocity_norm": (target_velocity**2).mean().item(),
        "velocity_norm": velocity_norm,
        "velocity_target_dot": velocity_target_dot,
        **target_metrics,
    }
    grad_terms = None
    if monitor_grad_norms:
        grad_terms = mse_vjp_surrogates(
            velocity,
            residual_terms,
            velocity.detach() - target_velocity.detach(),
        )
    return loss, metrics, grad_terms


def compute_sft_loss(model, latent, prompt_embeds, pooled_embeds, timestep, config, accelerator):
    """Reference-SFT loss on cached endpoints forward-noised to random timesteps."""
    host = adapter_host(model)
    host.set_adapter("default")

    sigma = (timestep / 1000).view(-1, 1, 1, 1)
    noise = torch.randn_like(latent)
    sample = (1 - sigma) * latent + sigma * noise

    velocity = model_forward(model, sample, timestep, prompt_embeds, pooled_embeds, accelerator)
    target_velocity = noise - latent
    per_sample = ((velocity - target_velocity.detach()) ** 2).mean(dim=tuple(range(1, velocity.ndim)))

    K = config.sft_num_loss_targets_per_sample
    if K > 1:
        assert per_sample.shape[0] % K == 0, "SFT loss batch size must be divisible by targets per sample"
        per_sample = per_sample.view(-1, K).mean(dim=1)
    loss = per_sample.mean()

    metrics = {
        "sft_target_velocity_norm": (target_velocity**2).mean().item(),
        "sft_output_delta_norm": ((velocity - target_velocity.detach()) ** 2).mean().item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Sampling, scoring, advantages
# ---------------------------------------------------------------------------


def compute_energy_advantages(
    pipe,
    prompt_embeds,
    pooled_embeds,
    latents,
    images,
    prompt_indices,
    neg_embeds,
    neg_pooled,
    config,
    accelerator,
    energy_scorer,
    epoch,
):
    """Generate optional base references and return energy advantages and metrics."""
    reference_latents = reference_images = None
    if config.energy_scale > 0.0:
        if prompt_indices is None or len(prompt_indices) != prompt_embeds.shape[0]:
            raise ValueError(
                "energy confinement needs one dataset prompt index for every prompt"
            )
        M = config.energy_reference_samples_per_prompt
        reference_embeds = prompt_embeds.repeat_interleave(M, 0)
        reference_pooled = pooled_embeds.repeat_interleave(M, 0)
        reference_seeds = [
            energy_scoring.reference_seed(config.seed, epoch, prompt_index, sample_index)
            for prompt_index in prompt_indices
            for sample_index in range(M)
        ]
        with adapter_host(pipe.transformer).disable_adapter():
            reference_latents, reference_images = generate_images(
                pipe,
                reference_embeds,
                reference_pooled,
                reference_seeds,
                neg_embeds,
                neg_pooled,
                config,
                accelerator,
                decode_images=energy_scorer.requires_images,
                allow_partial_batch=True,
            )
    return energy_scorer.compute_epoch(
        latents,
        images,
        config.num_samples_per_prompt,
        reference_latents=reference_latents,
        reference_images=reference_images,
    )


def sample_epoch(
    pipe, prompts, image_seeds, neg_embeds, neg_pooled, reward_fns, config, accelerator,
    epoch=0, dreamsim_scorers=None, fd_term=None, energy_scorer=None,
    prompt_indices=None,
):
    """Generate ``num_samples_per_prompt`` images per prompt, score them, and
    return per-sample group-relative advantages.

    ``fd_term`` (an ``fd_reward.FdTerm``) enables the Fréchet-distance-to-base
    term; it is passed when FD is on.
    """
    bs = sampling_batch_size(config, config.train_guidance_scale)
    N = config.num_samples_per_prompt * len(prompts)
    assert N % bs == 0, "num_samples_per_prompt × prompts_per_epoch (local) must be divisible by sampling_batch_size"
    if len(image_seeds) != len(prompts) or any(
        len(seeds) != config.num_samples_per_prompt for seeds in image_seeds
    ):
        raise ValueError(
            "image_seeds must contain num_samples_per_prompt seeds for every prompt"
        )
    with accelerator.autocast():
        prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
            prompts, prompt_2=None, prompt_3=None, do_classifier_free_guidance=False
        )
    embeds = prompt_embeds.repeat_interleave(config.num_samples_per_prompt, 0)
    pooled = pooled_embeds.repeat_interleave(config.num_samples_per_prompt, 0)
    prompts_flat = [p for p in prompts for _ in range(config.num_samples_per_prompt)]
    seeds_flat = []
    for seeds_for_prompt in image_seeds:
        seeds_flat.extend(seeds_for_prompt)

    latents, images = generate_images(
        pipe, embeds, pooled, seeds_flat, neg_embeds, neg_pooled, config, accelerator
    )

    # Per-reward scores
    per_reward = {name: [] for name, _ in reward_fns}
    for i in range(0, N, config.reward_batch_size):
        imgs = images[i:i + config.reward_batch_size]
        prms = prompts_flat[i:i + config.reward_batch_size]
        for name, fn in reward_fns:
            per_reward[name].append(fn(imgs, prms))
    reward_dict = {name: torch.cat(v) for name, v in per_reward.items()}
    rewards = sum(reward_dict.values())

    # Group-relative advantages
    G = config.num_samples_per_prompt
    gate_mode = config.dreamsim_gate
    # One gather of the (n,) task rewards, before the DreamSim and FD terms.
    # Its spread drives the epoch scaling and the DreamSim gate, and is logged
    # as reward_epoch_std in every run, whatever scale_rewards says.
    gathered_rewards = accelerator.gather(rewards)
    adv_scalars = {
        "reward_epoch_std": gathered_rewards.detach().float().cpu().std(correction=0).item(),
    }
    epoch_std = None
    if config.scale_rewards == "epoch" or gate_mode == "std":
        epoch_std = gathered_rewards.std(correction=0) + 1e-4

    # Task-reward spread per prompt group, taken before the DreamSim term is
    # added: it drives the DreamSim gate and the solved/unsolved logging.
    group_std = dreamsim_diversity.group_reward_std(rewards, G)
    zero_variance = dreamsim_diversity.is_zero_variance(group_std)
    group_mean_reward = rewards.view(-1, G).float().mean(dim=1)

    advantages = torch.zeros_like(rewards)
    for s in range(0, len(rewards), G):
        group = rewards[s:s + G]
        adv = group - group.mean()
        if config.scale_rewards == "group":
            adv = adv / (group.std(correction=0) + 1e-4)
        elif config.scale_rewards == "epoch":
            adv = adv / epoch_std
        elif config.scale_rewards not in (False, None, "none"):
            raise ValueError("scale_rewards must be 'group', 'epoch', or 'none'")
        advantages[s:s + G] = adv

    # DreamSim diversity: every configured variant is logged, and the variant
    # named by dreamsim_train_type optionally contributes each sample's centered
    # mean distance from its groupmates (zero-mean per group, then scaled by the
    # dreamsim_gate factor so it fades where the task reward has no spread).
    # Logging several variants while training on one shows whether a diversity
    # gain is real or specific to the optimized encoder; logging diversity
    # separately for solved and unsolved groups shows where the gate acts.
    dreamsim_tensors = {}
    masked_tensors = {"groups/zero_variance_mean_reward": (group_mean_reward, zero_variance)}
    if dreamsim_scorers is not None and dreamsim_diversity.dreamsim_is_active_epoch(config, epoch):
        scored = dreamsim_scorers.compute_epoch(images, G, latents=latents)
        for name, (diversity, model_advantages) in scored.items():
            dreamsim_tensors[f"dreamsim_diversity/{name}"] = diversity
            dreamsim_tensors[f"dreamsim_advantage_abs/{name}"] = model_advantages.abs()
            masked_tensors[f"dreamsim_diversity_solved/{name}"] = (diversity, zero_variance)
            masked_tensors[f"dreamsim_diversity_unsolved/{name}"] = (diversity, ~zero_variance)
        if dreamsim_diversity.dreamsim_reward_is_enabled(config):
            _, centered = scored[config.dreamsim_train_type]
            gate = dreamsim_diversity.gate_factors(group_std, gate_mode, epoch_std)
            gate_per_sample = dreamsim_diversity.expand_by_group(gate, G).to(advantages.dtype)
            dreamsim_advantages = config.dreamsim_scale * (
                gate_per_sample * centered.to(advantages.dtype)
            )
            advantages = advantages + dreamsim_advantages
            dreamsim_tensors["dreamsim_advantage_abs"] = dreamsim_advantages.abs()
            dreamsim_tensors["dreamsim_gate_mean"] = gate

    # Energy scoring-rule advantage. The helper owns frozen-base reference
    # generation; the additive contribution stays explicit here.
    energy_tensors = {}
    if energy_scorer is not None:
        energy_advantages, energy_tensors = compute_energy_advantages(
            pipe,
            prompt_embeds,
            pooled_embeds,
            latents,
            images,
            prompt_indices,
            neg_embeds,
            neg_pooled,
            config,
            accelerator,
            energy_scorer,
            epoch,
        )
        energy_advantages = energy_advantages.to(
            device=advantages.device, dtype=advantages.dtype
        )
        advantages = advantages + energy_advantages

    # Fréchet distance to the base model (fd_reward.py): the term embeds
    # the epoch's images, rewards them against the previous EMA state, and
    # updates the EMA.  Nothing happens on inactive epochs.
    fd_tensors = {}
    fd_scalars = {}
    fd_result = fd_term.compute_epoch(images, G, advantages, accelerator, epoch) if fd_term else None
    if fd_result is not None:
        if fd_result.advantage_delta is not None:
            advantages = advantages + fd_result.advantage_delta
        fd_tensors, fd_scalars = fd_result.tensors, fd_result.scalars

    # Tail of the FINAL advantages (task reward plus DreamSim, energy, and FD),
    # gathered so every rank reports the same number and reduced on CPU.
    gathered_abs_advantages = accelerator.gather(advantages).detach().cpu().float().abs()
    adv_scalars["adv_p99"] = torch.quantile(gathered_abs_advantages, 0.99).item()
    adv_scalars["adv_max"] = gathered_abs_advantages.max().item()

    metric_tensors = {
        "rewards": rewards,
        "abs_advantages": advantages.abs(),
        "groups/zero_variance_frac": zero_variance.float(),
    }
    metric_tensors.update({f"rewards/{name}": value for name, value in reward_dict.items()})
    metric_tensors.update(dreamsim_tensors)
    metric_tensors.update(energy_tensors)
    metric_tensors.update(fd_tensors)
    metrics = reduce_tensor_means(accelerator, metric_tensors)
    metrics.update(reduce_masked_means(accelerator, masked_tensors))
    metrics.update(fd_scalars)
    metrics.update(adv_scalars)
    return latents, embeds, pooled, advantages, metrics


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def cfg_label(guidance_scale):
    """Return a compact guidance-scale label suitable for a metric path."""
    return f"{float(guidance_scale):g}"


def validate(
    pipe,
    val_prompts,
    val_scales,
    val_prompt_indices,
    config,
    step,
    reward_fns,
    eval_guidance_scales,
    accelerator,
):
    """Generate rank-local samples and globally aggregate metrics by CFG."""
    if not (len(val_prompts) == len(val_scales) == len(val_prompt_indices)):
        raise ValueError("Validation prompts, scales, and prompt indices must have equal lengths")

    image_batches = []
    generated_prompts = []
    generated_scales = []
    for guidance_scale in eval_guidance_scales:
        scale_indices = [
            index for index, scale in enumerate(val_scales) if scale == guidance_scale
        ]
        bs = (
            config.sampling_batch_size // 2
            if guidance_scale > 1.0
            else config.sampling_batch_size
        )
        if bs <= 0:
            raise ValueError(
                f"sampling_batch_size={config.sampling_batch_size} is too small for CFG"
            )
        for i in range(0, len(scale_indices), bs):
            batch_indices = scale_indices[i:i + bs]
            batch_prompts = [val_prompts[index] for index in batch_indices]
            generators = [
                torch.Generator(device=pipe.device).manual_seed(
                    config.seed + val_prompt_indices[index]
                )
                for index in batch_indices
            ]
            with accelerator.autocast():
                imgs = pipe(
                    batch_prompts,
                    height=config.resolution,
                    width=config.resolution,
                    num_inference_steps=config.num_eval_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generators,
                    output_type="pt",
                    return_dict=False,
                )[0].float()
            image_batches.append(imgs)
            generated_prompts.extend(batch_prompts)
            generated_scales.extend([guidance_scale] * len(batch_prompts))

    images = torch.cat(image_batches) if image_batches else None

    def score(fn):
        if images is None:
            return torch.empty(0, device=pipe.device, dtype=torch.float32)
        chunks = [
            fn(
                images[i:i + config.reward_batch_size],
                generated_prompts[i:i + config.reward_batch_size],
            )
            for i in range(0, len(generated_prompts), config.reward_batch_size)
        ]
        return torch.cat(chunks)

    scores = {name: score(fn) for name, fn in reward_fns}
    if scores:
        rewards = sum(scores.values())
    else:
        rewards = torch.zeros(
            len(generated_prompts), device=pipe.device, dtype=torch.float32
        )
    metric_tensors = dict(scores)
    metric_tensors["rewards"] = rewards

    for name in config.validation_rewards:
        # Validation-only reward models are intentionally short lived. Move
        # them off the GPU explicitly before dropping the final reference;
        # relying on cyclic GC left ImageReward resident after validation and
        # could push the following SD3 training forward over 95 GiB.
        validation_reward = getattr(rewards_module, name)().to(pipe.device).eval()
        try:
            metric_tensors[name] = score(validation_reward)
        finally:
            validation_reward.to("cpu")
            del validation_reward
            release_cuda_memory()

    log = {}
    for guidance_scale in eval_guidance_scales:
        local_indices = [
            index
            for index, scale in enumerate(generated_scales)
            if scale == guidance_scale
        ]
        scale_tensors = {}
        for name, values in metric_tensors.items():
            indices = torch.tensor(local_indices, device=values.device, dtype=torch.long)
            scale_tensors[name] = values.index_select(0, indices)
        metrics = reduce_tensor_means(accelerator, scale_tensors)
        prefix = f"eval/cfg_{cfg_label(guidance_scale)}"
        log.update({f"{prefix}/{name}": value for name, value in metrics.items()})

    if accelerator.is_main_process:
        accelerator.log(log, step=step)

    # Preserve the original main-rank-only W&B image logging structure.
    if config.wandb_project and accelerator.is_main_process and images is not None:
        import wandb

        np_images = pipe.image_processor.pt_to_numpy(images)
        pil_images = pipe.image_processor.numpy_to_pil(np_images)
        logged_samples = {}
        for guidance_scale in eval_guidance_scales:
            local_indices = [
                index
                for index, scale in enumerate(generated_scales)
                if scale == guidance_scale
            ]
            key = (
                "samples"
                if len(eval_guidance_scales) == 1
                else f"samples/cfg_{cfg_label(guidance_scale)}"
            )
            logged_samples[key] = [
                wandb.Image(
                    pil_images[index],
                    caption=(
                        f"CFG={guidance_scale:g} "
                        f"{generated_prompts[index][:50]}... R={rewards[index].item():.2f}"
                    ),
                )
                for index in local_indices
            ]
        wandb.log(logged_samples, step=step)


def validate_dreamsim_diversity(
    pipe, diversity_prompts, diversity_prompt_indices, config, step,
    dreamsim_scorers, guidance_scale, accelerator, vendi_encoder=None,
):
    """Measure DreamSim and Inception Vendi diversity on the evaluation adapter.

    Generates ``dreamsim_eval_samples_per_prompt`` images for each rank-local
    prompt at a single CFG value and logs mean diversity per variant.  Unlike
    the training-time metric this reflects the lagged ``evaluation`` adapter at
    ``num_eval_inference_steps``, matching the policy the offline
    ``evaluation/`` pipeline scores.

    DreamSim diversity is a mean over pairs of i.i.d. generations, so its
    expectation is independent of the group size. Vendi Score is an effective
    sample count bounded by the group size, so train/eval values with different
    sample counts must be interpreted separately.
    """
    samples_per_prompt = config.dreamsim_eval_samples_per_prompt
    bs = (
        config.sampling_batch_size // 2
        if guidance_scale > 1.0
        else config.sampling_batch_size
    )
    if bs <= 0:
        raise ValueError(
            f"sampling_batch_size={config.sampling_batch_size} is too small for CFG"
        )

    # Whole prompt groups stay on one rank so diversity never crosses ranks.
    prompts_flat = [p for p in diversity_prompts for _ in range(samples_per_prompt)]
    seeds_flat = [
        dreamsim_diversity.eval_image_seed(config.seed, prompt_index, sample_index)
        for prompt_index in diversity_prompt_indices
        for sample_index in range(samples_per_prompt)
    ]

    image_batches = []
    latent_batches = []
    use_latents = dreamsim_scorers is not None and "latent" in dreamsim_scorers
    requires_images = (
        vendi_encoder is not None
        or (dreamsim_scorers is not None and dreamsim_scorers.requires_images)
    )
    for i in range(0, len(prompts_flat), bs):
        batch_prompts = prompts_flat[i:i + bs]
        generators = [
            torch.Generator(device=pipe.device).manual_seed(seed)
            for seed in seeds_flat[i:i + bs]
        ]
        with accelerator.autocast():
            output = pipe(
                batch_prompts,
                height=config.resolution,
                width=config.resolution,
                num_inference_steps=config.num_eval_inference_steps,
                guidance_scale=guidance_scale,
                generator=generators,
                output_type="latent" if use_latents else "pt",
                return_dict=False,
            )[0]
        if use_latents:
            latent_batches.append(output)
            if requires_images:
                lat = output / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
                decoded = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
                image_batches.append(pipe.image_processor.postprocess(decoded, output_type="pt").float())
        else:
            image_batches.append(output.float())

    device = pipe.device
    if image_batches or latent_batches:
        images = torch.cat(image_batches) if image_batches else None
        latents = torch.cat(latent_batches) if latent_batches else None
        metric_tensors = {}
        if dreamsim_scorers is not None:
            scored = dreamsim_scorers.compute_epoch(
                images, samples_per_prompt, with_advantages=False, latents=latents
            )
            metric_tensors.update({
                f"dreamsim_diversity/{name}": diversity
                for name, (diversity, _) in scored.items()
            })
        if vendi_encoder is not None:
            metric_tensors["vendi_score/inception_v3"] = grouped_vendi_scores(
                vendi_encoder.embed(images), samples_per_prompt
            )
    else:
        # A rank can receive no prompts when they do not divide evenly; it still
        # takes part in the reduction with an empty contribution.
        metric_tensors = {}
        if dreamsim_scorers is not None:
            metric_tensors.update({
                f"dreamsim_diversity/{name}": torch.empty(
                    0, device=device, dtype=torch.float32
                )
                for name in dreamsim_scorers.type_names
            })
        if vendi_encoder is not None:
            metric_tensors["vendi_score/inception_v3"] = torch.empty(
                0, device=device, dtype=torch.float32
            )

    metrics = reduce_tensor_means(accelerator, metric_tensors)
    if accelerator.is_main_process:
        accelerator.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)


# ---------------------------------------------------------------------------
# Pipeline build & checkpointing
# ---------------------------------------------------------------------------


def build_pipeline(config, accelerator):
    """Load SD3.5, freeze base parameters, attach the three LoRA adapters, and
    move modules to the right device + dtype.
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(config.model_name, cache_dir="models")
    pipe.set_progress_bar_config(disable=True)

    for m in [pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3, pipe.transformer]:
        m.requires_grad_(False)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=LORA_TARGET_MODULES,
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.add_adapter("old", lora_config)
    pipe.transformer.add_adapter("evaluation", lora_config)

    dtype = DTYPE_MAP[str(config.mixed_precision)]
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    for tx in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        tx.to(accelerator.device, dtype=dtype)
    pipe.transformer.to(accelerator.device)
    for name, p in pipe.transformer.named_parameters():
        if "lora_" in name:
            p.data = p.data.to(torch.float32)
        elif not p.requires_grad:
            p.data = p.data.to(dtype)
    assert all(
        p.dtype == torch.float32
        for name, p in pipe.transformer.named_parameters()
        if "lora_" in name
    )
    return pipe


class EpochState:
    """Tracks the epoch counter and accumulated wall time across checkpoints."""

    def __init__(self):
        self.epoch = -1
        self.total_train_seconds = 0.0

    def state_dict(self):
        return {"epoch": self.epoch, "total_train_seconds": self.total_train_seconds}

    def load_state_dict(self, d):
        self.epoch = d["epoch"]
        self.total_train_seconds = d.get("total_train_seconds", 0.0)


def save_checkpoint(accelerator, output_dir, epoch, epoch_state, total_seconds):
    """Save accelerate state to ``<output_dir>/step_<epoch>`` and update the
    ``latest`` symlink."""
    epoch_state.epoch = epoch
    epoch_state.total_train_seconds = total_seconds
    ckpt_dir = output_dir / f"step_{epoch}"
    accelerator.save_state(ckpt_dir)
    if accelerator.is_main_process:
        latest = output_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(ckpt_dir.resolve())
    accelerator.wait_for_everyone()
    accelerator.print(f"[Epoch {epoch}] Saved {ckpt_dir}")


def restore_checkpoint_rng(checkpoint_dir, process_index, fallback_seed):
    """Restore one rank's RNG streams after all stochastic startup work.

    Accelerate restores these as part of ``load_state``, but model and metric
    construction after that call may consume randomness. Reapply the saved
    state immediately before the training loop. A larger resumed world size
    has no files for its new ranks, so initialize those ranks deterministically.
    """
    rng_path = Path(checkpoint_dir) / f"random_states_{process_index}.pkl"
    if not rng_path.is_file():
        seed = int(fallback_seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        return False

    states = torch.load(rng_path, map_location="cpu", weights_only=False)
    random.setstate(states["random_state"])
    np.random.set_state(states["numpy_random_seed"])
    torch.set_rng_state(states["torch_manual_seed"])
    torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
    return True


# ---------------------------------------------------------------------------
# Per-epoch training step
# ---------------------------------------------------------------------------


def cycled_indices(order, start, count):
    positions = (torch.arange(count, device=order.device) + start) % len(order)
    return order[positions]


def build_sft_microbatch(sft_data, endpoint_indices, scheduler, config, accelerator):
    endpoint_indices_cpu = endpoint_indices.detach().cpu()
    prompt_indices = sft_data.endpoint_prompt_indices_cpu[endpoint_indices_cpu]

    latents = sft_data.latents_cpu[endpoint_indices_cpu].to(device=accelerator.device, non_blocking=True)
    embeds = sft_data.prompt_embeds_cpu[prompt_indices].to(device=accelerator.device, non_blocking=True)
    pooled = sft_data.pooled_embeds_cpu[prompt_indices].to(device=accelerator.device, non_blocking=True)

    K = config.sft_num_loss_targets_per_sample
    if K > 1:
        latents = latents.repeat_interleave(K, dim=0)
        embeds = embeds.repeat_interleave(K, dim=0)
        pooled = pooled.repeat_interleave(K, dim=0)

    timesteps = sample_timesteps(
        scheduler,
        config,
        num_samples=len(endpoint_indices_cpu),
        num_targets_per_sample=K,
        device=accelerator.device,
    )
    return latents, embeds, pooled, timesteps


def trainable_parameters(optimizer):
    """Yield the parameters the optimizer actually updates (the LoRA weights)."""
    for group in optimizer.param_groups:
        yield from group["params"]


def snapshot_trainable_params(optimizer):
    """Return CPU float32 copies of the trainable parameters.

    Host RAM only, not a single byte of device memory: each parameter is copied
    to the CPU first and only then widened to float32.  For SD3.5-medium at LoRA
    rank 32 one such copy is about 75 MB, and about 225 MB per rank is retained
    while an epoch runs (this snapshot, the delta, and the previous delta).
    """
    return [
        parameter.detach().to("cpu", copy=True).float()
        for parameter in trainable_parameters(optimizer)
    ]


def parameter_update_metrics(optimizer, snapshot, config):
    """Size and direction of one epoch's parameter update, computed on CPU.

    ``update_norm`` is the norm of the flattened epoch update, ``update_cos``
    its cosine with the previous epoch's update (NaN when there is none), and
    ``update_rel`` the norm divided by ``lr * sqrt(numel) * num_optim_steps_per_epoch``,
    i.e. Adam's average per-coordinate step per optimizer step as a fraction of
    ``lr``.  Under DDP every rank holds the same parameters after each step, so
    every rank computes the same numbers.

    The previous update hangs off the optimizer object and is never
    checkpointed, so the cosine is NaN on the first epoch of every segment.  A
    zero update (every fp16 step of the epoch skipped, or lr 0) is not stored,
    so it cannot make the next epoch's cosine NaN as well.
    """
    delta = torch.cat([
        parameter.detach().to("cpu", copy=True).float().sub_(before).reshape(-1)
        for parameter, before in zip(trainable_parameters(optimizer), snapshot, strict=True)
    ])
    norm = delta.norm().item()
    previous = getattr(optimizer, "_ram_prev_update", None)
    cos = float("nan")
    if previous is not None and previous.shape == delta.shape:
        denominator = norm * previous.norm().item()
        if denominator > 0.0:
            cos = torch.dot(delta, previous).item() / denominator
    if norm > 0.0:
        optimizer._ram_prev_update = delta
    scale = float(config.lr) * math.sqrt(delta.numel()) * float(config.num_optim_steps_per_epoch)
    return {
        "update_norm": norm,
        "update_cos": cos,
        "update_rel": norm / scale if scale > 0.0 else float("nan"),
    }


def train_one_epoch(
    model,
    optimizer,
    scheduler,
    latents,
    embeds,
    pooled,
    advantages,
    timesteps,
    config,
    accelerator,
    sft_data=None,
    sft_weight=0.0,
    grad_monitor=None,
):
    """Run one epoch of gradient-accumulated AdamW updates and return averaged
    metrics + the summed gradient norm across optim steps.
    """
    N = len(latents)
    bs = config.loss_batch_size
    assert N % bs == 0
    n_batches = N // bs

    use_sft = sft_data is not None and float(sft_weight) != 0.0
    if use_sft:
        sft_K = config.sft_num_loss_targets_per_sample
        assert bs % sft_K == 0
        sft_endpoint_order = torch.randperm(len(sft_data), device=accelerator.device)
        sft_endpoints_per_batch = bs // sft_K

    totals = {}
    grad_totals = {}
    num_grad_steps = 0
    grad_norm_sum = 0.0
    # CPU snapshot of the trainable parameters, for the epoch update below.
    param_snapshot = snapshot_trainable_params(optimizer)
    model.train()
    for batch_idx, i in enumerate(range(0, N, bs)):
        sync_context = grad_monitor.no_sync(model) if grad_monitor is not None else nullcontext()
        with accelerator.accumulate(model), sync_context:
            ram_loss, ram_metrics, ram_grad_terms = compute_main_loss(
                model, latents[i:i + bs], embeds[i:i + bs], pooled[i:i + bs],
                advantages[i:i + bs], timesteps[i:i + bs], config, accelerator,
                monitor_grad_norms=grad_monitor is not None,
            )
            metrics = dict(ram_metrics)
            metrics["ram_loss"] = ram_loss.item()
            total_loss = metrics["ram_loss"]
            if grad_monitor is not None:
                grad_monitor.capture(ram_grad_terms)
            accelerator.backward(ram_loss)
            del ram_loss, ram_grad_terms

            if use_sft:
                sft_endpoint_idx = cycled_indices(
                    sft_endpoint_order,
                    batch_idx * sft_endpoints_per_batch,
                    sft_endpoints_per_batch,
                )
                sft_latents, sft_embeds, sft_pooled, sft_timesteps = build_sft_microbatch(
                    sft_data, sft_endpoint_idx, scheduler, config, accelerator
                )
                sft_loss, sft_metrics = compute_sft_loss(
                    model,
                    sft_latents,
                    sft_embeds,
                    sft_pooled,
                    sft_timesteps,
                    config,
                    accelerator,
                )
                weighted_sft_loss = float(sft_weight) * sft_loss
                if grad_monitor is not None:
                    grad_monitor.capture({"sft_anchor": weighted_sft_loss})
                accelerator.backward(weighted_sft_loss)
                metrics.update(sft_metrics)
                metrics["sft_loss"] = sft_loss.item()
                metrics["sft_loss_weight"] = float(sft_weight)
                total_loss += float(sft_weight) * metrics["sft_loss"]
                del sft_latents, sft_embeds, sft_pooled, sft_timesteps, sft_loss, weighted_sft_loss

            metrics["loss"] = total_loss
            if accelerator.sync_gradients:
                if grad_monitor is None:
                    accelerator.unscale_gradients()
                    grad_norm_sum += torch.nn.utils.get_total_norm(
                        [p.grad for p in model.parameters() if p.grad is not None]
                    ).item()
                else:
                    step_grad_metrics = grad_monitor.finish(optimizer)
                    grad_norm_sum += step_grad_metrics["grad_norm/combined"]
                    for key, value in step_grad_metrics.items():
                        grad_totals[key] = grad_totals.get(key, 0.0) + value
                    num_grad_steps += 1
            optimizer.step()
            optimizer.zero_grad()
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
    model.eval()
    metrics = reduce_scalar_dict(accelerator, {k: v / n_batches for k, v in totals.items()})
    # Identical on every rank (DDP keeps the parameters in sync), so these are
    # merged after the reduce rather than averaged through it: an all-reduce of
    # the first epoch's NaN cosine would be a needless NaN round trip.
    metrics.update(parameter_update_metrics(optimizer, param_snapshot, config))
    del param_snapshot
    if grad_monitor is not None:
        assert num_grad_steps > 0
        grad_metrics = reduce_scalar_dict(
            accelerator,
            {key: value / num_grad_steps for key, value in grad_totals.items()},
        )
        metrics.update(grad_metrics)
    grad_norm_sum = reduce_scalar_dict(accelerator, {"grad_norm_sum": grad_norm_sum})["grad_norm_sum"]
    return metrics, grad_norm_sum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_prompts(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def derive_local_sizes(config, num_processes):
    """Compute per-process schedule sizes from the totals declared in config.

    The config specifies prompts per epoch as a *total* across all processes so
    that runs reproduce identically on any number of GPUs. Per-process batch
    sizes (``loss_batch_size``, ``sampling_batch_size``, ``reward_batch_size``)
    stay per-process because they reflect GPU memory.
    """
    if config.num_prompts_per_epoch % num_processes != 0:
        raise ValueError(
            f"num_prompts_per_epoch={config.num_prompts_per_epoch} must be divisible by "
            f"num_processes={num_processes}"
        )
    prompts_local = config.num_prompts_per_epoch // num_processes
    samples_local = prompts_local * config.num_samples_per_prompt
    if samples_local % config.sampling_batch_size != 0:
        raise ValueError(
            f"local samples per epoch ({samples_local}) must be divisible by "
            f"sampling_batch_size={config.sampling_batch_size}"
        )
    loss_samples_local = samples_local * config.num_loss_targets_per_sample
    optim_steps = config.num_optim_steps_per_epoch
    if loss_samples_local % (config.loss_batch_size * optim_steps) != 0:
        raise ValueError(
            f"local loss samples per epoch ({loss_samples_local}) must be divisible by "
            f"loss_batch_size × num_optim_steps_per_epoch = "
            f"{config.loss_batch_size} × {optim_steps}"
        )
    if config.loss_batch_size % config.num_loss_targets_per_sample != 0:
        raise ValueError(
            f"loss_batch_size={config.loss_batch_size} must be divisible by "
            f"num_loss_targets_per_sample={config.num_loss_targets_per_sample}"
        )
    if sft_is_enabled(config) and config.loss_batch_size % config.sft_num_loss_targets_per_sample != 0:
        raise ValueError(
            f"loss_batch_size={config.loss_batch_size} must be divisible by "
            f"sft_num_loss_targets_per_sample={config.sft_num_loss_targets_per_sample}"
        )
    grad_accum = loss_samples_local // (config.loss_batch_size * optim_steps)
    return prompts_local, samples_local, grad_accum


def main(config, output_dir, resume_from_checkpoint=None, stop_after_epoch=None):
    config = apply_main_loss_config(config)
    config = apply_wandb_config_defaults(config)
    config = apply_grad_norm_monitor_config_defaults(config)
    config = apply_reproducibility_config_defaults(config)
    config = apply_validation_config_defaults(config)
    config = apply_sft_config_defaults(config)
    config = dreamsim_diversity.apply_dreamsim_config_defaults(config)
    config = fd_reward.apply_fd_config_defaults(config)
    config = energy_scoring.apply_energy_config_defaults(config)
    sft_enabled = sft_is_enabled(config)
    fd_enabled = fd_reward.fd_is_enabled(config)
    # We need num_processes before constructing Accelerator (to compute
    # gradient_accumulation_steps from totals), so read it from the launcher.
    num_processes = int(os.environ.get("WORLD_SIZE", "1"))
    prompts_local, samples_local, grad_accum = derive_local_sizes(config, num_processes)
    eval_guidance_scales = resolve_eval_guidance_scales(config)
    if stop_after_epoch is not None and (
        stop_after_epoch <= 0 or stop_after_epoch % config.eval_every_n_epochs != 0
    ):
        raise ValueError("stop_after_epoch must be a positive validation/checkpoint epoch")

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=grad_accum,
        log_with="wandb" if config.wandb_project else None,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=1))],
    )
    assert accelerator.num_processes == num_processes, (
        f"WORLD_SIZE={num_processes} but Accelerator sees {accelerator.num_processes}"
    )

    # Output directory + config dump
    output_dir = Path(output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(vars(config), f)
    accelerator.wait_for_everyone()

    # Optional W&B logging
    if config.wandb_project:
        wandb_kwargs = {"name": resolve_wandb_run_name(config, output_dir)}
        if getattr(config, "wandb_entity", None):
            wandb_kwargs["entity"] = config.wandb_entity
        accelerator.init_trackers(
            project_name=config.wandb_project,
            config=vars(config),
            init_kwargs={"wandb": wandb_kwargs},
        )

    # Pipeline + adapters
    pipe = build_pipeline(config, accelerator)
    default_params, old_params, eval_params = collect_adapter_params(pipe.transformer)
    sync_lagged_params(default_params, old_params)
    sync_lagged_params(default_params, eval_params)

    optimizer = torch.optim.AdamW(
        default_params, lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay
    )
    model, optimizer = accelerator.prepare(pipe.transformer, optimizer)
    pipe.to(accelerator.device)

    # Resume
    epoch_state = EpochState()
    accelerator.register_for_checkpointing(epoch_state)
    # The FD spaces (one EMA per loaded space) ride along with the epoch state
    # as custom_checkpoint_1; registering only when FD is on keeps the
    # checkpoint layout of other runs.  Loading refuses a changed space set or
    # a different reference key.
    fd_spaces = None
    if fd_enabled:
        fd_key = fd_reference.fd_reference_key(fd_reference.fd_reference_metadata(config))
        fd_spaces = fd_reward.FdSpaces(config, fd_key)
        accelerator.register_for_checkpointing(fd_spaces)
    latest_link = output_dir / "latest"
    if resume_from_checkpoint is None:
        resume_checkpoint = latest_link if latest_link.is_symlink() else None
    else:
        resume_checkpoint = Path(resume_from_checkpoint).expanduser()
        if not resume_checkpoint.is_absolute():
            resume_checkpoint = (Path.cwd() / resume_checkpoint).resolve()
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(f"Resume checkpoint is not a directory: {resume_checkpoint}")
    start_epoch = 0
    resumed_checkpoint_epoch = None
    if resume_checkpoint is not None:
        if fd_enabled and not (resume_checkpoint / "custom_checkpoint_1.pkl").exists():
            raise RuntimeError(
                f"Checkpoint {resume_checkpoint.resolve()} predates the FD term (no FD state was "
                "saved); resume without FD or start a fresh output_dir"
            )
        accelerator.load_state(resume_checkpoint)
        # Checkpoints are written after validation but before this epoch's
        # optimization. Resume that same epoch and suppress only the validation
        # that is already represented by the checkpoint.
        start_epoch = epoch_state.epoch
        resumed_checkpoint_epoch = epoch_state.epoch
        if stop_after_epoch is not None and stop_after_epoch <= start_epoch:
            raise ValueError(
                f"stop_after_epoch={stop_after_epoch} must exceed checkpoint epoch {start_epoch}"
            )
        accelerator.print(
            f"Restored pre-optimization checkpoint {resume_checkpoint.resolve()}; "
            f"continuing at epoch {start_epoch}"
        )
        # `accelerator.load_state` restores the checkpoint's optimizer state, which
        # carries the lr, betas and weight_decay the run was checkpointed with, so a
        # resumed run would otherwise silently ignore a changed config value.  Re-apply
        # them from the config here: a no-op for a same-config resume, and there is no
        # LR scheduler in this script whose state could disagree.  `optimizer` is the
        # accelerate-prepared optimizer; its `param_groups` proxies to the wrapped torch
        # optimizer, which is the one that steps.
        # Capture what the checkpoint restored first, so the startup log proves a
        # branch-off actually changed a value instead of only echoing the config.
        before = [(g["lr"], g["betas"], g["weight_decay"]) for g in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["lr"] = config.lr
            group["betas"] = (config.beta1, config.beta2)
            group["weight_decay"] = config.weight_decay
        accelerator.print(
            f"Optimizer hyperparameters re-applied from config: lr={config.lr} "
            f"betas=({config.beta1}, {config.beta2}) weight_decay={config.weight_decay} "
            f"(checkpoint had lr={before[0][0]}, betas={before[0][1]}, "
            f"weight_decay={before[0][2]})"
        )
        if fd_enabled and not fd_spaces.is_initialized():
            raise RuntimeError(
                f"Checkpoint {resume_checkpoint.resolve()} predates the FD term (FD moments are empty); "
                "resume without FD or start a fresh output_dir"
            )
    set_adapter(model, "default")

    # Training schedules are built globally before prompts are sharded.
    train_prompts = load_prompts(Path("prompts") / config.prompts / "train.txt")
    val_prompts = load_prompts(Path("prompts") / config.prompts / "test.txt")
    if not val_prompts:
        raise ValueError("Validation prompt file must contain at least one prompt")
    num_val_prompts = max(1, math.floor(len(val_prompts) * config.eval_prompt_fraction))
    val_prompts = val_prompts[:num_val_prompts]

    # Preserve the original all-rank strided prompt shard. Each rank evaluates
    # its existing prompt shard at every CFG value.
    val_prompt_indices = list(range(num_val_prompts))[
        accelerator.process_index::num_processes
    ]
    val_prompts = [val_prompts[index] for index in val_prompt_indices]
    rank_val_prompts = val_prompts
    rank_val_prompt_indices = val_prompt_indices
    val_prompts = [
        prompt
        for _ in eval_guidance_scales
        for prompt in rank_val_prompts
    ]
    val_scales = [
        guidance_scale
        for guidance_scale in eval_guidance_scales
        for _ in rank_val_prompts
    ]
    val_prompt_indices = [
        prompt_index
        for _ in eval_guidance_scales
        for prompt_index in rank_val_prompt_indices
    ]

    # Validation-time diversity uses its own small prompt subset, taken from the
    # head of the full test set so it does not move when eval_prompt_fraction
    # changes, and sharded so each rank owns whole prompt groups.
    diversity_prompt_indices = []
    diversity_prompts = []
    vendi_eval_enabled = (
        fd_enabled
        and "inception_v3" in fd_reward.fd_loaded_types(config)
        and config.dreamsim_eval_num_prompts > 0
    )
    if dreamsim_diversity.dreamsim_eval_is_enabled(config) or vendi_eval_enabled:
        all_val_prompts = load_prompts(Path("prompts") / config.prompts / "test.txt")
        num_diversity_prompts = min(config.dreamsim_eval_num_prompts, len(all_val_prompts))
        diversity_prompt_indices = list(range(num_diversity_prompts))[
            accelerator.process_index::num_processes
        ]
        diversity_prompts = [all_val_prompts[index] for index in diversity_prompt_indices]

    train_sampler = PromptSeedBatchSampler(
        dataset_size=len(train_prompts),
        num_prompts_per_epoch=config.num_prompts_per_epoch,
        num_samples_per_prompt=config.num_samples_per_prompt,
        num_replicas=num_processes,
        rank=accelerator.process_index,
        seed=config.seed,
    )

    # Sampling noise uses sampler-owned generators; this stream covers the
    # remaining rank-local stochastic training operations.
    seed = config.seed + accelerator.process_index
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Reward modules
    reward_fns = [(name, getattr(rewards_module, name)().to(accelerator.device).eval()) for name in config.rewards]
    dreamsim_scorers = dreamsim_diversity.build_scorers(config, accelerator.device)

    # Negative prompt embeddings (used during sampling when CFG is on)
    with torch.no_grad(), accelerator.autocast():
        neg_embeds, _, neg_pooled, _ = pipe.encode_prompt(
            [""], prompt_2=None, prompt_3=None, do_classifier_free_guidance=False
        )

    model.eval()

    # FD reference statistics are rebuilt from the prepared cache on every
    # start (they are not checkpointed); each space's moments are warm-started
    # from the same features on a fresh run so G = 0 exactly at epoch zero.
    # Training never generates the reference: scripts/fd_prepare.py does, once.
    fd_term = None
    if fd_enabled:
        names = fd_spaces.names
        ref_features = fd_reference.load_reference_features(config, names)
        null_fd = fd_reference.load_null_fd(config, names)
        for name in names:
            features = ref_features.pop(name).to(accelerator.device)
            encoder = fd_encoders.FdEncoder(
                name, config.fd_encoder_cache_dir, config.fd_batch_size, accelerator.device
            )
            reference = fd_reward.FdReference.from_features(
                features, config.num_samples_per_prompt, config, cache_key=fd_key
            )
            fd_spaces.attach(name, encoder, reference, null_fd[name])
            if not fd_spaces[name].moments.is_initialized():
                fd_spaces[name].moments.warm_start(
                    features, config.num_samples_per_prompt, reference_key=fd_key
                )
            role = "train" if fd_spaces[name].trained else "log"
            accelerator.print(
                f"FD space {name} ({role}): {tuple(features.shape)} reference features, "
                f"null-FD {null_fd[name]:.4g}"
            )
            del features
        torch.cuda.empty_cache()
        fd_term = fd_reward.FdTerm(fd_spaces, config)

    energy_scorer = energy_scoring.build_scorer(
        config,
        accelerator.device,
        dreamsim_scorers=dreamsim_scorers,
        fd_spaces=fd_spaces,
    )
    if energy_scorer is not None:
        accelerator.print(
            "Energy reward: "
            f"features={config.energy_feature_extractor}, "
            f"normalize={config.energy_normalize_features}, beta={config.energy_beta:g}, "
            f"lambda={config.energy_lambda:g}, scale={config.energy_scale:g}, "
            f"references/prompt={config.energy_reference_samples_per_prompt}"
        )

    # Accelerate loaded the full checkpoint before the encoders to avoid a
    # transient memory peak. Reapply only the saved RNG streams after every
    # stochastic one-time setup step so startup cannot overwrite them.
    if resume_checkpoint is not None:
        rng_restored = restore_checkpoint_rng(
            resume_checkpoint,
            accelerator.process_index,
            config.seed + accelerator.process_index,
        )
        rng_flags = accelerator.gather(
            torch.tensor([int(rng_restored)], device=accelerator.device, dtype=torch.int64)
        )
        if accelerator.is_main_process:
            fallback_ranks = [index for index, flag in enumerate(rng_flags.tolist()) if not flag]
            if fallback_ranks:
                accelerator.print(
                    "No saved RNG state for new ranks "
                    f"{fallback_ranks}; initialized them from deterministic rank seeds"
                )
            else:
                accelerator.print("Restored checkpoint RNG state on every rank")
    set_adapter(model, "default")
    if fd_enabled:
        accelerator.print(
            f"FD reference {fd_reference.fd_reference_dir(config)}; normalize={config.fd_normalize}; "
            f"EMA age {fd_spaces.age}"
        )

    # ----- Training loop (runs until killed) -----
    total_seconds = epoch_state.total_train_seconds
    epoch = start_epoch
    while True:
        # Validation + checkpoint
        validation_due = epoch > 0 and epoch % config.eval_every_n_epochs == 0
        validation_already_checkpointed = epoch == resumed_checkpoint_epoch
        if validation_due and validation_already_checkpointed:
            accelerator.print(
                f"[Epoch {epoch}] Validation and checkpoint already completed; "
                "continuing with optimization"
            )
        elif validation_due:
            accelerator.print(f"[Epoch {epoch}] Validation...")
            set_adapter(model, "evaluation")
            with torch.no_grad():
                release_cuda_memory()
                validate(
                    pipe,
                    val_prompts,
                    val_scales,
                    val_prompt_indices,
                    config,
                    epoch,
                    reward_fns,
                    eval_guidance_scales,
                    accelerator,
                )
                # Release decoded validation images and the temporary quality
                # scorer before beginning the diversity pass.
                release_cuda_memory()
                if (
                    dreamsim_diversity.dreamsim_eval_is_enabled(config)
                    or vendi_eval_enabled
                ):
                    validate_dreamsim_diversity(
                        pipe,
                        diversity_prompts,
                        diversity_prompt_indices,
                        config,
                        epoch,
                        dreamsim_scorers,
                        dreamsim_diversity.eval_guidance_scale(config, eval_guidance_scales),
                        accelerator,
                        vendi_encoder=(
                            fd_spaces["inception_v3"].encoder
                            if vendi_eval_enabled else None
                        ),
                    )
                    release_cuda_memory()
            set_adapter(model, "default")
            if accelerator.is_main_process:
                accelerator.log({"eval/gpu_hours": total_seconds / 3600 * num_processes}, step=epoch)
            save_checkpoint(accelerator, output_dir, epoch, epoch_state, total_seconds)
            if stop_after_epoch is not None and epoch == stop_after_epoch:
                accelerator.print(f"[Epoch {epoch}] Reached requested checkpoint; stopping")
                break
            # Checkpoint and W&B objects can also participate in Python cycles.
            # The next loss forward runs close to GH200 capacity.
            release_cuda_memory()

        # Sample, score, build advantages
        t_start = time.time()
        sft_weight = resolve_sft_loss_weight(config, epoch) if sft_enabled else 0.0
        train_sampler.set_epoch(epoch)
        prompt_seed_batch = next(iter(train_sampler))

        prompts = []
        image_seeds = []
        prompt_indices = []
        for prompt_index, seeds_for_prompt in prompt_seed_batch:
            prompts.append(train_prompts[prompt_index])
            image_seeds.append(seeds_for_prompt)
            prompt_indices.append(prompt_index)

        assert len(prompts) == prompts_local
        set_adapter(model, "old")
        with torch.no_grad():
            latents, embeds, pooled, advantages, sample_metrics = sample_epoch(
                pipe, prompts, image_seeds, neg_embeds, neg_pooled, reward_fns, config, accelerator,
                epoch=epoch, dreamsim_scorers=dreamsim_scorers, fd_term=fd_term,
                energy_scorer=energy_scorer, prompt_indices=prompt_indices,
            )
        set_adapter(model, "default")
        latents = latents.detach()
        if "fd_ema_age" in sample_metrics:
            if epoch == start_epoch:
                fd_keys = sorted(key for key in sample_metrics if key.startswith("fd"))
                accelerator.print("FD metrics under train/: " + ", ".join(fd_keys))
            for name in fd_spaces.names:
                accelerator.print(
                    f"[Epoch {epoch}] fd[{name}]={sample_metrics[f'fd/{name}']:.4g} "
                    f"fd_resid={sample_metrics[f'fd_resid/{name}']:.4g} "
                    f"fd_ratio={sample_metrics[f'fd_ratio/{name}']:.4g} "
                    f"spread_ratio={sample_metrics[f'fd_reward_spread_ratio/{name}']:.4g} "
                    f"cond_S={sample_metrics[f'fd_cond_S/{name}']:.3g} "
                    f"clamp_frac={sample_metrics[f'fd_clamp_frac/{name}']:.3g}"
                )
            accelerator.print(
                f"[Epoch {epoch}] fd combined: spread_ratio={sample_metrics['fd_reward_spread_ratio']:.4g} "
                f"ratio_mean={sample_metrics['fd_ratio_mean']:.4g} "
                f"rank_diff={sample_metrics['fd_rank_max_diff']:.3g} "
                f"age={sample_metrics['fd_ema_age']:.0f}"
                + (
                    f" fd_adv_abs={sample_metrics['fd_advantage_abs']:.4g}"
                    if "fd_advantage_abs" in sample_metrics else ""
                )
            )

        sft_data = None
        if sft_enabled and sft_weight != 0.0:
            with torch.no_grad():
                sft_data = sft_cache.sample_sft_epoch(
                    pipe, prompts, neg_embeds, neg_pooled, config, accelerator, adapter_host
                )
            sample_metrics.update(reduce_scalar_dict(accelerator, sft_data.metrics, reduction="sum"))
        if accelerator.is_main_process:
            accelerator.log({f"train/{k}": v for k, v in sample_metrics.items()}, step=epoch)

        # Permute samples (when num_optim_steps > 1) and tile by loss-targets
        if config.num_optim_steps_per_epoch > 1:
            order = torch.randperm(len(latents))
            latents, embeds, pooled, advantages = latents[order], embeds[order], pooled[order], advantages[order]
        K = config.num_loss_targets_per_sample
        if K > 1:
            latents = latents.repeat_interleave(K, dim=0)
            embeds = embeds.repeat_interleave(K, dim=0)
            pooled = pooled.repeat_interleave(K, dim=0)
            advantages = advantages.repeat_interleave(K, dim=0)

        timesteps = sample_timesteps(
            pipe.scheduler, config,
            num_samples=samples_local,
            num_targets_per_sample=K,
            device=accelerator.device,
        )

        grad_monitor = None
        if should_monitor_epoch(config.grad_norm_monitor_every_n_epochs, epoch):
            optimizer_parameters = (
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
            grad_monitor = GradNormMonitor(optimizer_parameters, accelerator)

        train_metrics, grad_norm_sum = train_one_epoch(
            model, optimizer, pipe.scheduler, latents, embeds, pooled, advantages, timesteps, config, accelerator,
            sft_data=sft_data, sft_weight=sft_weight, grad_monitor=grad_monitor,
        )
        if grad_monitor is not None:
            grad_monitor.release()

        # EMA updates of the lagged adapters
        update_lagged_params(default_params, old_params, epoch + 1, config.ema_decay, config.ema_warmup_rate)
        update_lagged_params(default_params, eval_params, epoch + 1, 0.9, None)

        total_seconds += reduce_max(accelerator, time.time() - t_start)
        log = {f"train/{k}": v for k, v in train_metrics.items()}
        if sft_enabled and "train/sft_loss_weight" not in log:
            log["train/sft_loss_weight"] = sft_weight
        log["train/grad_norm"] = grad_norm_sum / config.num_optim_steps_per_epoch
        log["train/gpu_hours"] = total_seconds / 3600 * num_processes
        if accelerator.is_main_process:
            accelerator.log(log, step=epoch)
        accelerator.print(f"[Epoch {epoch}] loss={log['train/loss']:.4f}")
        epoch_state.epoch = epoch
        epoch_state.total_train_seconds = total_seconds
        epoch += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_overrides(extra_args):
    """Parse trailing ``--key value`` pairs into a dict (values via YAML)."""
    out = {}
    for i in range(0, len(extra_args), 2):
        out[extra_args[i].lstrip("-")] = yaml.safe_load(extra_args[i + 1])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="config name (without .yaml)")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="output directory; default outputs/<config>. Re-running with the same dir resumes from latest.",
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="explicit checkpoint directory to load; otherwise use <output_dir>/latest when present.",
    )
    parser.add_argument(
        "--stop_after_epoch", type=int, default=None,
        help="stop successfully immediately after saving this validation checkpoint epoch",
    )
    args, overrides = parser.parse_known_args()

    config = load_config(args.config)
    config.update(parse_overrides(overrides))
    output_dir = args.output_dir or f"outputs/{args.config}"
    main(argparse.Namespace(**config), output_dir, args.resume_from_checkpoint, args.stop_after_epoch)
