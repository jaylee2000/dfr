"""DreamSim diversity: epoch-level logging and a group-centered reward term.

DreamSim diversity for a prompt is the mean pairwise perceptual distance among
the ``N`` images generated for that prompt, averaged over prompts::

    Diversity_k = 2 / (N (N - 1)) * sum_{i < j} d(g_i^k, g_j^k)

where ``d = (1 - cosine_similarity) / 2`` is DreamSim's normalized perceptual
distance in ``[0, 1]``. The offline ``evaluation/`` DreamSim metric uses the
unnormalized cosine distance, so divide its values by two before comparison.

The ``latent`` variant uses final denoised latents before VAE scaling/shifting,
flattened and unit-normalized, with ``d = ||u_i - u_j||^2 / 4``. It requires
no image encoder and shares the reward averaging, centering, and gating below.

Diversity is a mean over pairs of i.i.d. generations, so its expectation is
``E[d(g, g')]`` regardless of ``N`` -- only its variance shrinks as ``N`` grows.
Diversity numbers from different group sizes are therefore comparable, which is
what lets validation use a cheaper ``N`` than training.

The per-image reward is its mean distance to the other generations for the
same prompt::

    Reward_p = 1 / (N - 1) * sum_{j != p} d(g_p, g_j)

Subtracting the prompt-group mean gives a zero-mean DreamSim advantage. A
positive ``dreamsim_scale`` therefore favors samples farther from their
groupmates without changing the prompt group's mean advantage.

Once a prompt's group is *solved* -- every sample earns the same task reward --
the task advantage is exactly zero and DreamSim would be the only signal left:
an unbounded push to spread siblings apart along whatever the task reward does
not price (layout, object scale, background, style).  ``dreamsim_gate`` scales
the term per group by the task reward's within-group spread ``std_g`` so it
fades together with the task signal instead of replacing it::

    none    factor = 1                                  (unchanged behaviour)
    solved  factor = 1[std_g > 0]                       (cliff)
    std     factor = min(1, std_g / epoch_std)          (slope)

Under ``scale_rewards: epoch`` the task advantage in group ``g`` has magnitude
about ``std_g / epoch_std``, so the ``std`` gate holds the DreamSim-to-task
ratio constant across groups, and the clamp keeps high-variance groups at the
ungated strength so ``dreamsim_scale`` calibrations carry over.  The gate is
computed from the task reward alone, before the DreamSim term is added, and it
only touches the DreamSim term; the task advantage and the KL anchors are
unchanged.

Because this reward is a group-level quantity it cannot be expressed through the
per-sample ``reward_models.Reward`` interface, so it lives here rather than in
``reward_models``.
"""

import hashlib
import math

import torch
from torch import nn

# Image side length DreamSim's own preprocessing resizes to.  Read from the
# package when available so the two stay in sync.
try:  # pragma: no cover - depends on the dreamsim install
    from dreamsim.config import dreamsim_args as _dreamsim_args

    DREAMSIM_IMAGE_SIZE = int(_dreamsim_args["img_size"])
except Exception:  # pragma: no cover
    DREAMSIM_IMAGE_SIZE = 224

# A pairwise diversity reward requires at least one other sample in each group.
MIN_SAMPLES_PER_PROMPT = 2

# Selectable DreamSim variants, mapped to the (dreamsim_type, use_patch_model)
# pair ``dreamsim()`` expects.  The patch checkpoints must be requested through
# ``use_patch_model`` rather than by name: passing "dino_vitb16_patch" as
# dreamsim_type resolves the single-branch weight tag instead of the patch one.
DREAMSIM_TYPES = {
    "ensemble": ("ensemble", False),
    "dino_vitb16": ("dino_vitb16", False),
    "dino_vitb16_patch": ("dino_vitb16", True),
    "dinov2_vitb14": ("dinov2_vitb14", False),
    "dinov2_vitb14_patch": ("dinov2_vitb14", True),
    "clip_vitb32": ("clip_vitb32", False),
    "open_clip_vitb32": ("open_clip_vitb32", False),
    "synclr_vitb16": ("synclr_vitb16", False),
}

DEFAULT_DREAMSIM_TYPE = "dino_vitb16_patch"
DIVERSITY_TYPES = (*DREAMSIM_TYPES, "latent")

# How the DreamSim term is scaled per prompt group; see the module docstring.
DREAMSIM_GATES = ("none", "solved", "std")

# Population std of the task reward at or below which a group counts as
# zero-variance ("solved").  GenEval scores are exact fractions, so identical
# samples give exactly zero; the tolerance only guards float accumulation.
ZERO_VARIANCE_TOL = 1e-6


def resolve_type(name):
    """Map a config variant name to ``(dreamsim_type, use_patch_model)``."""
    if not isinstance(name, str) or name not in DREAMSIM_TYPES:
        raise ValueError(
            f"Unknown DreamSim variant {name!r}; choose from {sorted(DREAMSIM_TYPES)}"
        )
    return DREAMSIM_TYPES[name]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _validate_diversity_type(name):
    if not isinstance(name, str) or name not in DIVERSITY_TYPES:
        raise ValueError(f"Unknown diversity variant {name!r}; choose from {sorted(DIVERSITY_TYPES)}")


def _validate_types(names, key):
    if isinstance(names, str) or not isinstance(names, (list, tuple)):
        raise ValueError(f"{key} must be a list of diversity variant names, got {names!r}")
    resolved = []
    for name in names:
        _validate_diversity_type(name)
        if name not in resolved:
            resolved.append(name)
    return resolved


def apply_dreamsim_config_defaults(config):
    """Fill in and validate the ``dreamsim_*`` config keys."""
    defaults = {
        "dreamsim_scale": 0.0,
        "dreamsim_train_type": DEFAULT_DREAMSIM_TYPE,
        "dreamsim_gate": "none",
        "dreamsim_log_types": [],
        "dreamsim_log_every_n_epochs": 0,
        "dreamsim_eval_num_prompts": 0,
        "dreamsim_eval_samples_per_prompt": None,
        "dreamsim_eval_guidance_scale": None,
        "dreamsim_cache_dir": "models/dreamsim",
        "dreamsim_batch_size": None,
    }
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)

    if config.dreamsim_scale is None:
        config.dreamsim_scale = 0.0
    if isinstance(config.dreamsim_scale, bool):
        raise ValueError(f"dreamsim_scale must be a finite number, got {config.dreamsim_scale!r}")
    config.dreamsim_scale = float(config.dreamsim_scale)
    if not math.isfinite(config.dreamsim_scale):
        raise ValueError(f"dreamsim_scale must be finite, got {config.dreamsim_scale!r}")

    _validate_diversity_type(config.dreamsim_train_type)

    gate = config.dreamsim_gate
    if gate is None:
        gate = "none"
    if not isinstance(gate, str) or gate not in DREAMSIM_GATES:
        raise ValueError(f"dreamsim_gate must be one of {DREAMSIM_GATES}, got {gate!r}")
    config.dreamsim_gate = gate

    config.dreamsim_log_types = _validate_types(
        config.dreamsim_log_types or [], "dreamsim_log_types"
    )

    interval = config.dreamsim_log_every_n_epochs
    if interval is None:
        interval = 0
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise ValueError(
            f"dreamsim_log_every_n_epochs must be an integer >= 0, got {interval!r}"
        )
    config.dreamsim_log_every_n_epochs = interval

    num_eval_prompts = config.dreamsim_eval_num_prompts
    if num_eval_prompts is None:
        num_eval_prompts = 0
    if isinstance(num_eval_prompts, bool) or not isinstance(num_eval_prompts, int) or num_eval_prompts < 0:
        raise ValueError(
            f"dreamsim_eval_num_prompts must be an integer >= 0, got {num_eval_prompts!r}"
        )
    config.dreamsim_eval_num_prompts = num_eval_prompts

    if config.dreamsim_eval_samples_per_prompt is None:
        config.dreamsim_eval_samples_per_prompt = config.num_samples_per_prompt
    config.dreamsim_eval_samples_per_prompt = int(config.dreamsim_eval_samples_per_prompt)

    if config.dreamsim_batch_size is None:
        config.dreamsim_batch_size = config.reward_batch_size
    config.dreamsim_batch_size = int(config.dreamsim_batch_size)
    if config.dreamsim_batch_size <= 0:
        raise ValueError("dreamsim_batch_size must be positive")

    if not config.dreamsim_cache_dir:
        raise ValueError("dreamsim_cache_dir must be a non-empty path")

    if dreamsim_reward_is_enabled(config) and config.num_samples_per_prompt < MIN_SAMPLES_PER_PROMPT:
        raise ValueError(
            "The DreamSim reward term needs num_samples_per_prompt >= "
            f"{MIN_SAMPLES_PER_PROMPT}, got {config.num_samples_per_prompt}"
        )
    if dreamsim_train_logging_is_enabled(config) and config.num_samples_per_prompt < 2:
        raise ValueError(
            "DreamSim diversity needs num_samples_per_prompt >= 2, got "
            f"{config.num_samples_per_prompt}"
        )
    if dreamsim_eval_is_enabled(config) and config.dreamsim_eval_samples_per_prompt < 2:
        raise ValueError(
            "DreamSim diversity needs dreamsim_eval_samples_per_prompt >= 2, got "
            f"{config.dreamsim_eval_samples_per_prompt}"
        )
    return config


def dreamsim_reward_is_enabled(config):
    """True when the centered mean-distance term is added to the advantages."""
    return float(config.dreamsim_scale) != 0.0


def dreamsim_train_logging_is_enabled(config):
    """True when diversity is measured during ``sample_epoch``."""
    return bool(config.dreamsim_log_types) and config.dreamsim_log_every_n_epochs > 0


def dreamsim_eval_is_enabled(config):
    """True when diversity is measured on the evaluation adapter."""
    return bool(config.dreamsim_log_types) and config.dreamsim_eval_num_prompts > 0


def dreamsim_is_enabled(config):
    """True when DreamSim is needed at all."""
    return (
        dreamsim_reward_is_enabled(config)
        or dreamsim_train_logging_is_enabled(config)
        or dreamsim_eval_is_enabled(config)
    )


def dreamsim_is_active_epoch(config, epoch):
    """True when diversity should be computed in ``sample_epoch`` for ``epoch``.

    When the reward term is on, its advantage changes training and must be computed
    every epoch; ``dreamsim_log_every_n_epochs`` then has no effect and the
    metrics are logged every epoch because they are already paid for.
    """
    if dreamsim_reward_is_enabled(config):
        return True
    interval = config.dreamsim_log_every_n_epochs
    return bool(config.dreamsim_log_types) and interval > 0 and epoch % interval == 0


def loaded_types(config):
    """Variant names to instantiate: the logged ones plus the trained-on one.

    The trained-on variant is always included so a run logs what it optimizes.
    """
    names = list(config.dreamsim_log_types)
    if dreamsim_reward_is_enabled(config) and config.dreamsim_train_type not in names:
        names.insert(0, config.dreamsim_train_type)
    return names


def eval_guidance_scale(config, eval_guidance_scales):
    """Resolve the single CFG value used for validation-time diversity."""
    if config.dreamsim_eval_guidance_scale is not None:
        return float(config.dreamsim_eval_guidance_scale)
    return float(eval_guidance_scales[0])


def eval_image_seed(seed, prompt_index, sample_index):
    """Deterministic per-image seed on a stream separate from training seeds."""
    payload = f"{seed}:dreamsim_eval:{prompt_index}:{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class DreamSimScorer(nn.Module):
    """Embeds ``[0, 1]`` image tensors with one DreamSim variant."""

    def __init__(self, type_name, cache_dir, batch_size, device):
        super().__init__()
        from dreamsim import dreamsim

        dreamsim_type, use_patch_model = resolve_type(type_name)
        self.type_name = type_name
        self.model, _ = dreamsim(
            dreamsim_type=dreamsim_type,
            use_patch_model=use_patch_model,
            device=str(device),
            cache_dir=str(cache_dir),
        )
        self.model.eval().requires_grad_(False)
        self.batch_size = int(batch_size)
        self.device = device

    def _resize(self, images):
        # DreamSim's own preprocessing is Resize(224, BICUBIC) + ToTensor() with
        # no normalization, so for images already in [0, 1] a bicubic resize is
        # the whole transform.  Done on-device to avoid a PIL round-trip.
        size = DREAMSIM_IMAGE_SIZE
        if images.shape[-2:] == (size, size):
            return images
        return nn.functional.interpolate(
            images, size=(size, size), mode="bicubic", align_corners=False, antialias=True
        ).clamp(0.0, 1.0)

    @staticmethod
    def _reduce(embeddings, normalize=True):
        """Reduce raw DreamSim features to one vector per image.

        Patch variants return ``(B, 1 + P, D)``; DreamSim's own distance
        concatenates the CLS token with the spatially averaged patch tokens
        before taking the cosine distance (see ``PerceptualModel.forward``).
        Every other variant already returns ``(B, D)`` and passes straight
        through the normalization.
        """
        if embeddings.ndim == 3 and embeddings.shape[1] > 1:
            cls_embedding = embeddings[:, 0, :]
            patch_embeddings = embeddings[:, 1:, :]
            patch_count = patch_embeddings.shape[1]
            patch_side = int(round(patch_count**0.5))
            if patch_side * patch_side == patch_count:
                patch_embedding = patch_embeddings.reshape(
                    embeddings.shape[0], patch_side, patch_side, embeddings.shape[-1]
                ).mean(dim=(1, 2))
                embeddings = torch.cat((cls_embedding, patch_embedding), dim=-1)
            else:
                embeddings = embeddings.flatten(start_dim=1)
        elif embeddings.ndim != 2:
            embeddings = embeddings.flatten(start_dim=1)
        if normalize:
            embeddings = embeddings / embeddings.norm(
                p=2, dim=-1, keepdim=True
            ).clamp_min(1e-12)
        return embeddings

    @torch.no_grad()
    def embed(self, images, normalize=True):
        """Embed images, optionally applying DreamSim's final L2 normalization."""
        chunks = []
        for start in range(0, images.shape[0], self.batch_size):
            batch = images[start:start + self.batch_size].to(self.device, dtype=torch.float32)
            features = self.model.embed(self._resize(batch))
            if features.ndim == 2 and batch.shape[0] == 1:
                # embed() squeezes singleton dims; restore the batch axis.
                features = features.unsqueeze(0)
            chunks.append(self._reduce(features, normalize=normalize).float())
        return torch.cat(chunks) if chunks else images.new_zeros((0, 1))

    @torch.no_grad()
    def compute_epoch(self, images, num_samples_per_prompt, with_advantages=True):
        """Score images laid out as contiguous per-prompt groups.

        Args:
            images: ``(num_prompts * N, 3, H, W)`` floats in ``[0, 1]``, ordered
                so that each prompt's ``N`` samples are contiguous.
            num_samples_per_prompt: group size ``N``.
            with_advantages: compute centered per-image advantages.

        Returns:
            ``(diversity, advantages)`` where ``diversity`` has one entry per
            prompt and ``advantages`` one per image in input order (``None``
            when ``with_advantages`` is false).
        """
        group_size = int(num_samples_per_prompt)
        if group_size < MIN_SAMPLES_PER_PROMPT:
            raise ValueError(
                "num_samples_per_prompt must be >= "
                f"{MIN_SAMPLES_PER_PROMPT}, got {group_size}"
            )
        total = images.shape[0]
        if total % group_size != 0:
            raise ValueError(
                f"number of images ({total}) must be divisible by "
                f"num_samples_per_prompt ({group_size})"
            )

        embeddings = self.embed(images).view(total // group_size, group_size, -1)

        # Normalized pairwise cosine distances; the diagonal is exactly zero.
        distances = (
            1.0 - torch.bmm(embeddings, embeddings.transpose(1, 2))
        ) / 2.0
        distances = distances - torch.diag_embed(torch.diagonal(distances, dim1=1, dim2=2))

        rewards = distances.sum(dim=2) / float(group_size - 1)
        diversity = rewards.mean(dim=1)
        if not with_advantages:
            return diversity, None

        advantages = rewards - diversity.unsqueeze(1)
        return diversity, advantages.reshape(-1)


class LatentScorer(nn.Module):
    """Squared distances between unit-normalized terminal latents, without an encoder."""

    @torch.no_grad()
    def compute_epoch(self, latents, num_samples_per_prompt, with_advantages=True):
        group_size = int(num_samples_per_prompt)
        if group_size < MIN_SAMPLES_PER_PROMPT:
            raise ValueError(f"num_samples_per_prompt must be >= {MIN_SAMPLES_PER_PROMPT}, got {group_size}")
        if latents.ndim < 2:
            raise ValueError("latents must have a batch dimension and feature dimensions")
        total = latents.shape[0]
        if total % group_size:
            raise ValueError(f"number of latents ({total}) must be divisible by num_samples_per_prompt ({group_size})")
        # Disabling autocast keeps the Gram matrix in float32 as well as the norms.
        with torch.autocast(device_type=latents.device.type, enabled=False):
            vectors = nn.functional.normalize(latents.float().flatten(1), dim=1, eps=1e-12)
            vectors = vectors.reshape(total // group_size, group_size, vectors.shape[-1])
            norms = vectors.square().sum(dim=-1)
            distances = (
                norms.unsqueeze(2) + norms.unsqueeze(1)
                - 2.0 * torch.bmm(vectors, vectors.transpose(1, 2))
            ).div(4.0).clamp(0.0, 1.0)
            distances.diagonal(dim1=1, dim2=2).zero_()
            rewards = distances.sum(dim=2) / (group_size - 1)
            diversity = rewards.mean(dim=1)
            advantages = (rewards - diversity.unsqueeze(1)).reshape(-1) if with_advantages else None
        return diversity, advantages


class DreamSimScorers:
    """A deduplicated set of image and latent diversity scorers."""

    def __init__(self, type_names, cache_dir, batch_size, device):
        self.scorers = {
            name: LatentScorer() if name == "latent" else DreamSimScorer(name, cache_dir, batch_size, device)
            for name in dict.fromkeys(type_names)
        }

    def __contains__(self, name):
        return name in self.scorers

    @property
    def type_names(self):
        return list(self.scorers)

    @property
    def requires_images(self):
        return any(name != "latent" for name in self.scorers)

    @torch.no_grad()
    def compute_epoch(self, images, num_samples_per_prompt, with_advantages=True, *, latents=None):
        """Score aligned images/terminal latents; images may be None for latent-only scoring."""
        if "latent" in self and latents is None:
            raise ValueError("latent diversity requires terminal latents")
        if self.requires_images and images is None:
            raise ValueError("DreamSim image encoders require images")
        if images is not None and latents is not None and len(images) != len(latents):
            raise ValueError("images and latents must have matching batch sizes")
        return {
            name: scorer.compute_epoch(
                latents if name == "latent" else images,
                num_samples_per_prompt,
                with_advantages=with_advantages,
            )
            for name, scorer in self.scorers.items()
        }


def build_scorers(config, device):
    """Instantiate the configured variants, or ``None`` when DreamSim is unused."""
    names = loaded_types(config)
    if not dreamsim_is_enabled(config) or not names:
        return None
    return DreamSimScorers(names, config.dreamsim_cache_dir, config.dreamsim_batch_size, device)


def center_by_group(values, group_size):
    """Subtract each contiguous group's mean, preserving the input shape."""
    grouped = values.view(-1, group_size)
    return (grouped - grouped.mean(dim=1, keepdim=True)).reshape(-1)


# ---------------------------------------------------------------------------
# Gating on the task reward
# ---------------------------------------------------------------------------


def group_reward_std(rewards, group_size):
    """Population std of the task reward within each contiguous group.

    Returns one value per group.  Call this on the task reward *before* the
    DreamSim term is added so the gate never sees its own contribution.
    """
    return rewards.view(-1, group_size).float().std(dim=1, correction=0)


def is_zero_variance(group_std):
    """Boolean mask of groups whose task reward does not vary ("solved")."""
    return group_std <= ZERO_VARIANCE_TOL


def gate_factors(group_std, mode, epoch_std=None):
    """Per-group multiplier in ``[0, 1]`` for the DreamSim advantage.

    Args:
        group_std: ``(num_groups,)`` task-reward std per group.
        mode: one of ``DREAMSIM_GATES``.
        epoch_std: scalar std of the task reward over the whole epoch (already
            offset away from zero); required for ``mode="std"``.
    """
    if mode == "none":
        return torch.ones_like(group_std)
    if mode == "solved":
        return (~is_zero_variance(group_std)).to(group_std.dtype)
    if mode == "std":
        if epoch_std is None:
            raise ValueError("gate_factors(mode='std') needs epoch_std")
        return (group_std / epoch_std).clamp(min=0.0, max=1.0)
    raise ValueError(f"dreamsim_gate must be one of {DREAMSIM_GATES}, got {mode!r}")


def expand_by_group(values, group_size):
    """Repeat one value per group into one value per sample, in group order."""
    return values.repeat_interleave(group_size)
