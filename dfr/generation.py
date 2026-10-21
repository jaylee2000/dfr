"""Shared image generation for training-time sampling and the FD reference pass.

``generate_images`` is the exact loop ``sample_epoch`` used to inline: it runs
the SD3 pipeline to terminal latents in sampling minibatches, decodes them with
the VAE, and returns both.  The FD reference pass (``fd_prepare.py``) calls
the same function with the adapters disabled, so the base-model samples cannot
drift from the training samples in any detail of the call.
"""

import torch


def sampling_batch_size(config, guidance_scale):
    """Per-process generation batch size; CFG doubles the transformer batch."""
    bs = config.sampling_batch_size // 2 if guidance_scale > 1.0 else config.sampling_batch_size
    if bs <= 0:
        raise ValueError(f"sampling_batch_size={config.sampling_batch_size} is too small for CFG")
    return bs


def generate_images(
    pipe,
    embeds,
    pooled,
    seeds,
    neg_embeds,
    neg_pooled,
    config,
    accelerator,
    *,
    decode_images=True,
    allow_partial_batch=False,
):
    """Generate one image per (prompt embedding, seed) at the training settings.

    Args:
        embeds, pooled: per-sample prompt embeddings, ``(N, ...)``.
        seeds: ``N`` integer seeds, one generator per image.
        neg_embeds, neg_pooled: the empty-prompt embeddings used when
            ``train_guidance_scale > 1``.

    Returns:
        ``(latents, images)``: terminal latents ``(N, C, h, w)`` and decoded
        images ``(N, 3, H, W)`` as floats in ``[0, 1]``. ``images`` is ``None``
        when ``decode_images`` is false.
    """
    N = embeds.shape[0]
    if len(seeds) != N:
        raise ValueError(f"expected {N} seeds, got {len(seeds)}")
    bs = sampling_batch_size(config, config.train_guidance_scale)
    if not allow_partial_batch and N % bs != 0:
        raise ValueError(f"number of samples ({N}) must be divisible by the sampling batch size ({bs})")

    all_latents, all_images = [], []
    for i in range(0, N, bs):
        batch_embeds = embeds[i : i + bs]
        batch_pooled = pooled[i : i + bs]
        if config.train_guidance_scale > 1.0:
            batch_neg_embeds = neg_embeds.expand(batch_embeds.shape[0], -1, -1)
            batch_neg_pooled = neg_pooled.expand(batch_pooled.shape[0], -1)
        else:
            batch_neg_embeds = batch_neg_pooled = None

        generators = [torch.Generator(device=pipe.device).manual_seed(seed) for seed in seeds[i : i + bs]]

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
                generator=generators,
                output_type="latent",
                return_dict=False,
            )[0]

        all_latents.append(terminal_latent)
        if decode_images:
            lat = terminal_latent / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            decoded = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
            images = pipe.image_processor.postprocess(decoded, output_type="pt")
            all_images.append(images)

    return torch.cat(all_latents), (torch.cat(all_images) if decode_images else None)
