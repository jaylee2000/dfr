"""Energy-kernel scoring-rule rewards with unbiased leave-one-out baselines.

For each prompt, let ``X_1, ..., X_N`` be policy samples and let
``Y_1, ..., Y_M`` be independent samples from the frozen base model.  This
module minimizes the population loss

    L = E[rho(X, Y)] - lambda / 2 E[rho(X, X')],

where ``rho(x, y) = ||f(x) - f(y)||_2**beta``.  The factor one half in the
interaction objective accounts for both policy-dependent endpoints.  The
negative loss is used as a score-function reward, and ``energy_scale``
multiplies its complete advantage.

Ordinary within-group mean subtraction biases a finite-sample REINFORCE
estimate because the baseline contains the sample being scored.  ``advantages``
uses the algebraically equivalent leave-one-out corrections separately:

    A_int  = scale * lambda * N/(N-2) * (r_int - mean(r_int))
    A_conf = -scale * N/(N-1) * (r_conf - mean(r_conf)).

Thus ``energy_lambda`` controls the interaction-to-confinement ratio while
``energy_scale`` controls the magnitude of the full energy advantage,
independently of ``N``.  Reference samples do not need an ``M`` correction:
they are independent of the policy score and are averaged directly.
"""

import hashlib
import math

import torch
from torch import nn

import dreamsim_diversity
import fd_encoders


def resolve_feature_extractor(value):
    """Return ``(family, name)`` for an energy feature selector."""
    if value == "latent":
        return "latent", None
    if not isinstance(value, str) or ":" not in value:
        raise ValueError("energy_feature_extractor must be 'latent', 'dreamsim:<variant>', " "or 'fd:<encoder>'")
    family, name = value.split(":", 1)
    if family == "dreamsim":
        dreamsim_diversity.resolve_type(name)
    elif family == "fd":
        fd_encoders.resolve_fd_type(name)
    else:
        raise ValueError("energy_feature_extractor must be 'latent', 'dreamsim:<variant>', " "or 'fd:<encoder>'")
    return family, name


def _finite_nonnegative(value, key):
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a finite non-negative number, got {value!r}")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a finite non-negative number, got {value!r}") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be a finite non-negative number, got {value!r}")
    return value


def _nonnegative_int(value, key):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be an integer >= 0, got {value!r}")
    return value


def apply_energy_config_defaults(config):
    """Fill and validate the additive ``energy_*`` reward configuration."""
    legacy_keys = [
        key
        for key in ("energy_interaction_scale", "energy_confinement_scale")
        if hasattr(config, key)
    ]
    if legacy_keys:
        names = ", ".join(legacy_keys)
        raise ValueError(
            f"Legacy energy parameters are no longer supported: {names}. "
            "Use energy_lambda and energy_scale instead."
        )
    defaults = {
        "energy_feature_extractor": "latent",
        "energy_normalize_features": True,
        "energy_beta": 1.0,
        "energy_lambda": 1.0,
        "energy_scale": 0.0,
        "energy_reference_samples_per_prompt": 0,
        "energy_batch_size": None,
    }
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)

    resolve_feature_extractor(config.energy_feature_extractor)
    if not isinstance(config.energy_normalize_features, bool):
        raise ValueError(
            "energy_normalize_features must be true or false, got " f"{config.energy_normalize_features!r}"
        )
    if isinstance(config.energy_beta, bool):
        raise ValueError(f"energy_beta must be in (0, 2], got {config.energy_beta!r}")
    try:
        config.energy_beta = float(config.energy_beta)
    except (TypeError, ValueError) as error:
        raise ValueError(f"energy_beta must be in (0, 2], got {config.energy_beta!r}") from error
    if not math.isfinite(config.energy_beta) or not 0.0 < config.energy_beta <= 2.0:
        raise ValueError(f"energy_beta must be in (0, 2], got {config.energy_beta!r}")

    config.energy_lambda = _finite_nonnegative(config.energy_lambda, "energy_lambda")
    config.energy_scale = _finite_nonnegative(config.energy_scale, "energy_scale")
    config.energy_reference_samples_per_prompt = _nonnegative_int(
        config.energy_reference_samples_per_prompt, "energy_reference_samples_per_prompt"
    )
    if config.energy_batch_size is None:
        config.energy_batch_size = config.reward_batch_size
    if (
        isinstance(config.energy_batch_size, bool)
        or not isinstance(config.energy_batch_size, int)
        or config.energy_batch_size <= 0
    ):
        raise ValueError("energy_batch_size must be a positive integer")

    if energy_is_enabled(config):
        n = config.num_samples_per_prompt
        if n < 2:
            raise ValueError("The energy advantage needs num_samples_per_prompt >= 2, " f"got {n}")
        if config.energy_lambda > 0.0 and n < 3:
            raise ValueError("The unbiased interaction advantage needs num_samples_per_prompt >= 3, " f"got {n}")
        if config.energy_reference_samples_per_prompt < 1:
            raise ValueError(
                "energy_reference_samples_per_prompt must be >= 1 when " "energy_scale is positive"
            )
    return config


def energy_is_enabled(config):
    return float(config.energy_scale) > 0.0


def reference_seed(seed, epoch, prompt_index, sample_index):
    """Stable base-reference seed, disjoint from policy and validation streams."""
    payload = f"{seed}:energy_reference:{epoch}:{prompt_index}:{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _normalize(features):
    return nn.functional.normalize(features, p=2, dim=-1, eps=1e-12)


def pairwise_energy(left, right, beta):
    """Raw ``||left-right||**beta`` matrix in float32."""
    with torch.autocast(device_type=left.device.type, enabled=False):
        distances = torch.cdist(left.float(), right.float(), p=2)
        if beta != 1.0:
            distances = distances.pow(beta)
    return distances


def reward_components(policy_features, group_size, beta, reference_features=None):
    """Return per-policy-sample interaction and confinement rewards.

    Features are laid out as contiguous prompt groups.  The diagonal of the
    interaction matrix is explicitly zeroed to avoid numerical self-distance.
    """
    n = int(group_size)
    if n < 2:
        raise ValueError(f"group_size must be >= 2, got {n}")
    if policy_features.ndim != 2 or policy_features.shape[0] % n:
        raise ValueError("policy features must be a 2D tensor divisible into prompt groups")
    groups = policy_features.reshape(-1, n, policy_features.shape[-1])
    interactions = []
    confinements = [] if reference_features is not None else None

    if reference_features is not None:
        if reference_features.ndim != 2 or reference_features.shape[0] % groups.shape[0]:
            raise ValueError("reference features must be a 2D tensor with the same number of prompt groups")
        m = reference_features.shape[0] // groups.shape[0]
        if m < 1:
            raise ValueError("each prompt needs at least one reference feature")
        references = reference_features.reshape(groups.shape[0], m, -1)
        if references.shape[-1] != groups.shape[-1]:
            raise ValueError("policy and reference features must have the same dimension")

    for group_index, group in enumerate(groups):
        distances = pairwise_energy(group, group, beta)
        distances.diagonal().zero_()
        interactions.append(distances.sum(dim=1) / float(n - 1))
        if reference_features is not None:
            cross = pairwise_energy(group, references[group_index], beta)
            confinements.append(cross.mean(dim=1))
    return torch.cat(interactions), (torch.cat(confinements) if confinements is not None else None)


def advantage_components(interaction_rewards, confinement_rewards, group_size, energy_lambda):
    """Return unscaled interaction and confinement terms with finite-``N`` corrections."""
    n = int(group_size)
    if interaction_rewards.ndim != 1 or interaction_rewards.shape[0] % n:
        raise ValueError("interaction_rewards must be divisible into prompt groups")
    interaction = interaction_rewards.reshape(-1, n)
    interaction_result = torch.zeros_like(interaction)
    confinement_result = torch.zeros_like(interaction)
    if energy_lambda != 0.0:
        if n < 3:
            raise ValueError("the unbiased interaction advantage requires group_size >= 3")
        centered = interaction - interaction.mean(dim=1, keepdim=True)
        interaction_result.add_(centered, alpha=float(energy_lambda) * n / (n - 2))
    if n < 2:
        raise ValueError("the unbiased confinement advantage requires group_size >= 2")
    if confinement_rewards is None or confinement_rewards.shape != interaction_rewards.shape:
        raise ValueError("confinement rewards must match interaction rewards")
    confinement = confinement_rewards.reshape(-1, n)
    centered = confinement - confinement.mean(dim=1, keepdim=True)
    confinement_result.add_(centered, alpha=-n / (n - 1))
    return interaction_result.reshape(-1), confinement_result.reshape(-1)


def unbiased_advantages(interaction_rewards, confinement_rewards, group_size, energy_lambda, energy_scale):
    """Combine the two terms using exact finite-``N`` LOO corrections."""
    interaction, confinement = advantage_components(
        interaction_rewards, confinement_rewards, group_size, energy_lambda
    )
    return float(energy_scale) * (interaction + confinement)


class EnergyScorer:
    """Feature extraction and energy advantages for one configured space."""

    def __init__(self, config, encoder=None):
        self.family, self.name = resolve_feature_extractor(config.energy_feature_extractor)
        self.encoder = encoder
        self.normalize_features = config.energy_normalize_features
        self.beta = config.energy_beta
        self.energy_lambda = config.energy_lambda
        self.energy_scale = config.energy_scale

    @property
    def requires_images(self):
        return self.family != "latent"

    @torch.no_grad()
    def embed(self, *, latents=None, images=None):
        if self.family == "latent":
            if latents is None:
                raise ValueError("latent energy scoring requires terminal latents")
            features = latents.float().flatten(start_dim=1)
        else:
            if images is None:
                raise ValueError(f"{self.family} energy scoring requires decoded images")
            if self.family == "dreamsim":
                features = self.encoder.embed(images, normalize=False)
            else:
                features = self.encoder.embed(images)
            features = features.float().flatten(start_dim=1)
        return _normalize(features) if self.normalize_features else features

    @torch.no_grad()
    def compute_epoch(self, policy_latents, policy_images, group_size, reference_latents=None, reference_images=None):
        policy = self.embed(latents=policy_latents, images=policy_images)
        reference = None
        if self.energy_scale != 0.0:
            reference = self.embed(latents=reference_latents, images=reference_images)
        interaction, confinement = reward_components(policy, group_size, self.beta, reference_features=reference)
        interaction_advantage, confinement_advantage = advantage_components(
            interaction,
            confinement,
            group_size,
            self.energy_lambda,
        )
        advantages = float(self.energy_scale) * (interaction_advantage + confinement_advantage)
        interaction_advantage = float(self.energy_scale) * interaction_advantage
        confinement_advantage = float(self.energy_scale) * confinement_advantage
        tensors = {
            "energy/interaction_reward": interaction,
            "energy/interaction_advantage_abs": interaction_advantage.abs(),
            "energy/advantage_abs": advantages.abs(),
        }
        if confinement is not None:
            tensors["energy/confinement_reward"] = confinement
            tensors["energy/confinement_advantage_abs"] = confinement_advantage.abs()
        return advantages, tensors


def build_scorer(config, device, dreamsim_scorers=None, fd_spaces=None):
    """Build the configured scorer, reusing an already-loaded encoder."""
    if not energy_is_enabled(config):
        return None
    family, name = resolve_feature_extractor(config.energy_feature_extractor)
    encoder = None
    if family == "dreamsim":
        if dreamsim_scorers is not None and name in dreamsim_scorers:
            encoder = dreamsim_scorers.scorers[name]
        else:
            encoder = dreamsim_diversity.DreamSimScorer(
                name, config.dreamsim_cache_dir, config.energy_batch_size, device
            )
    elif family == "fd":
        if fd_spaces is not None and name in fd_spaces.names and fd_spaces[name].encoder is not None:
            encoder = fd_spaces[name].encoder
        else:
            encoder = fd_encoders.FdEncoder(name, config.fd_encoder_cache_dir, config.energy_batch_size, device)
    return EnergyScorer(config, encoder=encoder)


__all__ = [
    "EnergyScorer",
    "advantage_components",
    "apply_energy_config_defaults",
    "build_scorer",
    "energy_is_enabled",
    "pairwise_energy",
    "reference_seed",
    "resolve_feature_extractor",
    "reward_components",
    "unbiased_advantages",
]
