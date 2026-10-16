"""Fréchet distance to the base model as a policy-gradient reward.

The Fréchet distance (FD) between the fine-tuned generator and the base model,
measured in a frozen encoder's feature space, depends on the policy only
through the raw feature moments ``mu_g = E[f]`` and ``M_g = E[f f^T]``.  The
score-function identity therefore turns its gradient into an ordinary policy
gradient whose per-sample reward is a scalar quadratic form in the feature::

    r_p = -[ w_mean * 2 (mu_g - mu_r)^T f_p  +  w_cov * (f_p - mu_g)^T G (f_p - mu_g) ]
    G   = I - Sigma_r^{1/2} (Sigma_r^{1/2} Sigma_g Sigma_r^{1/2})^{-1/2} Sigma_r^{1/2}

with the moments held in a per-epoch EMA warm-started from the base-model
features.  No gradients ever flow through the encoder, and one ``d x d``
eigendecomposition per epoch replaces a per-sample jackknife.  The numerics
are float64, with shrinkage (and its exact-gradient correction) and a clamp
on the spectrum of ``G``; the residual within-prompt variant carries a ``1/N``
correction.

Layout, mirroring ``dreamsim_diversity.py``: config handling first, the pure
float64 math (unit-testable on CPU) next, and ``FdTerm``, the per-epoch driver
``sample_epoch`` calls, last.  The encoders live in ``fd_encoders.py``.
"""

import math
from dataclasses import dataclass

import torch
from accelerate.utils import broadcast

from distributed_metrics import reduce_max
from dreamsim_diversity import center_by_group
from fd_encoders import DEFAULT_FD_TYPE, resolve_fd_type
from vendi_diversity import grouped_vendi_scores

FD_COV_SOURCES = ("residual", "pooled")
FD_NORMALIZE_MODES = ("none", "sg_ratio", "null_ratio")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _finite_float(value, key, minimum=None, exclusive_minimum=None, exclusive_maximum=None):
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a finite number, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite, got {value!r}")
    if minimum is not None and result < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value!r}")
    if exclusive_minimum is not None and result <= exclusive_minimum:
        raise ValueError(f"{key} must be > {exclusive_minimum}, got {value!r}")
    if exclusive_maximum is not None and result >= exclusive_maximum:
        raise ValueError(f"{key} must be < {exclusive_maximum}, got {value!r}")
    return result


def _int_at_least(value, key, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}, got {value!r}")
    return value


def _validate_type_list(names, key):
    if isinstance(names, str) or not isinstance(names, (list, tuple)):
        raise ValueError(f"{key} must be a list of FD encoder names, got {names!r}")
    resolved = []
    for name in names:
        resolve_fd_type(name)
        if name in resolved:
            raise ValueError(f"{key} names {name!r} twice")
        resolved.append(name)
    return resolved


def apply_fd_config_defaults(config):
    """Fill in and validate the ``fd_*`` configuration keys."""
    if hasattr(config, "fd_train_type"):
        raise ValueError(
            "fd_train_type has been replaced by fd_train_types, a list of registry names"
        )
    defaults = {
        "fd_train_types": [DEFAULT_FD_TYPE],
        "fd_train_weights": None,
        "fd_normalize": "null_ratio",
        "fd_norm_eps": None,
        "fd_log_types": [],
        "fd_scale": 0.0,
        # Both off unless a config asks for them: a per-sample clip on the
        # combined reward and a cap on the dose it adds to the advantage.
        "fd_sample_clip": None,
        "fd_realized_max": None,
        "fd_mean_weight": 1.0,
        "fd_cov_weight": 1.0,
        "fd_cov_source": "residual",
        "fd_ema_beta": 0.95,
        "fd_init_epochs": 25,
        "fd_shrinkage": 0.02,
        "fd_clamp": 10.0,
        "fd_eig_floor": 0.0,
        # Off unless a config asks for it, like dreamsim_log_every_n_epochs, so
        # existing configs are unaffected.  A nonzero fd_scale forces every epoch.
        "fd_log_every_n_epochs": 0,
        "fd_cache_dir": "models/fd_stats",
        "fd_encoder_cache_dir": "models/fd_encoders",
        "fd_batch_size": None,
        "fd_reference_seed": None,
        # Smoke tests only: relax the sample-count check on the reference pass.
        # Refused when the reward term is on.
        "fd_allow_small_reference": False,
    }
    for key, value in defaults.items():
        if not hasattr(config, key) or (key.endswith("_types") and getattr(config, key) is None):
            setattr(config, key, value)

    config.fd_train_types = _validate_type_list(config.fd_train_types, "fd_train_types")
    if not config.fd_train_types:
        raise ValueError("fd_train_types must name at least one FD encoder")
    config.fd_log_types = _validate_type_list(config.fd_log_types, "fd_log_types")
    if config.fd_train_weights is None:
        config.fd_train_weights = [1.0] * len(config.fd_train_types)
    if isinstance(config.fd_train_weights, str) or not isinstance(config.fd_train_weights, (list, tuple)):
        raise ValueError(f"fd_train_weights must be a list of numbers, got {config.fd_train_weights!r}")
    if len(config.fd_train_weights) != len(config.fd_train_types):
        raise ValueError(
            f"fd_train_weights has {len(config.fd_train_weights)} entries for "
            f"{len(config.fd_train_types)} fd_train_types"
        )
    config.fd_train_weights = [
        _finite_float(w, f"fd_train_weights[{i}]", minimum=0.0) for i, w in enumerate(config.fd_train_weights)
    ]
    if all(w == 0.0 for w in config.fd_train_weights):
        raise ValueError("fd_train_weights must not all be zero")
    if config.fd_normalize not in FD_NORMALIZE_MODES:
        raise ValueError(f"fd_normalize must be one of {FD_NORMALIZE_MODES}, got {config.fd_normalize!r}")
    if config.fd_norm_eps is not None:
        config.fd_norm_eps = _finite_float(config.fd_norm_eps, "fd_norm_eps", minimum=0.0)

    if config.fd_scale is None:
        config.fd_scale = 0.0
    config.fd_scale = _finite_float(config.fd_scale, "fd_scale")
    if config.fd_sample_clip is not None:
        config.fd_sample_clip = _finite_float(
            config.fd_sample_clip, "fd_sample_clip", exclusive_minimum=0.0
        )
    if config.fd_realized_max is not None:
        config.fd_realized_max = _finite_float(
            config.fd_realized_max, "fd_realized_max", exclusive_minimum=0.0
        )
    config.fd_mean_weight = _finite_float(config.fd_mean_weight, "fd_mean_weight", minimum=0.0)
    config.fd_cov_weight = _finite_float(config.fd_cov_weight, "fd_cov_weight", minimum=0.0)
    if config.fd_mean_weight == 0.0 and config.fd_cov_weight == 0.0:
        raise ValueError("fd_mean_weight and fd_cov_weight must not both be zero")
    if config.fd_cov_source not in FD_COV_SOURCES:
        raise ValueError(f"fd_cov_source must be one of {FD_COV_SOURCES}, got {config.fd_cov_source!r}")
    config.fd_ema_beta = _finite_float(
        config.fd_ema_beta, "fd_ema_beta", exclusive_minimum=0.0, exclusive_maximum=1.0
    )
    # The null-FD splits the reference epochs into two halves.
    config.fd_init_epochs = _int_at_least(config.fd_init_epochs, "fd_init_epochs", 2)
    config.fd_shrinkage = _finite_float(
        config.fd_shrinkage, "fd_shrinkage", minimum=0.0, exclusive_maximum=1.0
    )
    if config.fd_clamp is not None:
        config.fd_clamp = _finite_float(config.fd_clamp, "fd_clamp", exclusive_minimum=0.0)
    config.fd_eig_floor = _finite_float(
        config.fd_eig_floor, "fd_eig_floor", minimum=0.0, exclusive_maximum=1.0
    )
    if config.fd_log_every_n_epochs is None:
        config.fd_log_every_n_epochs = 0
    config.fd_log_every_n_epochs = _int_at_least(
        config.fd_log_every_n_epochs, "fd_log_every_n_epochs", 0
    )
    if config.fd_batch_size is None:
        config.fd_batch_size = config.reward_batch_size
    config.fd_batch_size = _int_at_least(int(config.fd_batch_size), "fd_batch_size", 1)
    if not config.fd_cache_dir:
        raise ValueError("fd_cache_dir must be a non-empty path")
    if not config.fd_encoder_cache_dir:
        raise ValueError("fd_encoder_cache_dir must be a non-empty path")
    if config.fd_reference_seed is not None:
        config.fd_reference_seed = _int_at_least(config.fd_reference_seed, "fd_reference_seed", 0)
    config.fd_allow_small_reference = bool(config.fd_allow_small_reference)

    if fd_is_enabled(config):
        if config.fd_cov_source == "residual" and config.num_samples_per_prompt < 2:
            raise ValueError(
                "fd_cov_source: residual needs num_samples_per_prompt >= 2, got "
                f"{config.num_samples_per_prompt}"
            )
        reference_samples = (
            config.fd_init_epochs * config.num_prompts_per_epoch * config.num_samples_per_prompt
        )
        widest = max(fd_loaded_types(config), key=lambda name: resolve_fd_type(name).dim)
        dim = resolve_fd_type(widest).dim
        if reference_samples < 2 * dim:
            if config.fd_allow_small_reference and not fd_reward_is_enabled(config):
                pass
            elif config.fd_allow_small_reference:
                raise ValueError(
                    "fd_allow_small_reference is only for measure-only smoke tests and is "
                    "refused when fd_scale != 0"
                )
            else:
                raise ValueError(
                    "The FD reference pass needs at least 2 x d samples: fd_init_epochs x "
                    "num_prompts_per_epoch x num_samples_per_prompt = "
                    f"{reference_samples} < 2 x {dim} for {widest}. "
                    "Raise fd_init_epochs."
                )
    return config


def fd_loaded_types(config):
    """Trained spaces first, then the log-only ones; each name once."""
    names = list(config.fd_train_types)
    for name in config.fd_log_types:
        if name not in names:
            names.append(name)
    return names


def fd_reward_is_enabled(config):
    """True when the FD term is added to the advantages."""
    return float(config.fd_scale) != 0.0


def fd_logging_is_enabled(config):
    """True when FD is measured during ``sample_epoch`` without entering the advantage."""
    return config.fd_log_every_n_epochs > 0


def fd_is_enabled(config):
    """True when the encoder, the reference pass and the EMA are needed at all."""
    return fd_reward_is_enabled(config) or fd_logging_is_enabled(config)


def fd_is_active_epoch(config, epoch):
    """True when the FD term is computed in ``sample_epoch`` for ``epoch``.

    With a nonzero scale the term changes training and is computed every
    epoch; ``fd_log_every_n_epochs`` then has no effect.  On inactive epochs
    nothing is embedded and the EMA is not updated.
    """
    if fd_reward_is_enabled(config):
        return True
    interval = config.fd_log_every_n_epochs
    return interval > 0 and epoch % interval == 0


# ---------------------------------------------------------------------------
# Pure math (float64, device-agnostic)
# ---------------------------------------------------------------------------


def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _raw_moments(features):
    """Mean and raw second moment of ``(n, d)`` float64 features."""
    n = features.shape[0]
    if n == 0:
        raise ValueError("Cannot take moments of zero samples")
    mu = features.mean(dim=0)
    second = _symmetrize(features.T @ features / n)
    return mu, second


def residual_features(features, group_size):
    """Subtract each contiguous group's mean feature, preserving the shape."""
    n, d = features.shape
    group_size = int(group_size)
    if group_size < 1 or n % group_size != 0:
        raise ValueError(
            f"number of samples ({n}) must be a positive multiple of the group size ({group_size})"
        )
    grouped = features.view(n // group_size, group_size, d)
    return (grouped - grouped.mean(dim=1, keepdim=True)).reshape(n, d)


def prepare_covariance(sigma, shrinkage, eig_floor):
    """Floor, then shrink, a symmetric PSD covariance in its own eigenbasis.

    The floor is relative to the largest eigenvalue (numerics doc, 3.3,
    placement B) and the shrinkage is ``(1 - rho) Sigma + rho (tr Sigma / d) I``
    (N.2).  Both preserve the eigenvectors, so one eigensolve also yields the
    symmetric square root.

    Returns ``(sigma_tilde, sigma_tilde_half, raw_eigenvalues, floor_fraction)``.
    """
    sigma = _symmetrize(sigma.to(torch.float64))
    lam_raw, basis = torch.linalg.eigh(sigma)
    lam = lam_raw
    floor_frac = 0.0
    if eig_floor > 0.0:
        floor_value = eig_floor * lam.max().clamp_min(0.0)
        floor_frac = (lam < floor_value).double().mean().item()
        lam = lam.clamp_min(floor_value)
    tau = lam.mean()
    lam_tilde = (1.0 - shrinkage) * lam + shrinkage * tau
    sigma_tilde = _symmetrize((basis * lam_tilde) @ basis.T)
    sigma_tilde_half = _symmetrize((basis * lam_tilde.clamp_min(0.0).sqrt()) @ basis.T)
    return sigma_tilde, sigma_tilde_half, lam_raw, floor_frac


@dataclass
class BuresGradient:
    """``G_eff`` for one covariance pair plus the diagnostics of its eigensolves."""

    G: torch.Tensor
    eigvals: torch.Tensor  # of the clamped G_eff
    eigvecs: torch.Tensor
    fd_cov: float
    cond_S: float
    G_min_eig: float  # before the clamp
    G_min_eig_ref_var: float
    clamp_frac: float


def bures_gradient(sigma_g_tilde, ref_tilde, ref_tilde_half, ref_raw, shrinkage, clamp):
    """Gradient matrix of the Bures term between shrunk covariances.

    Implements Sections 3.2 and 3.3 of the plan: ``S = R^{1/2} Sigma_g R^{1/2}``,
    ``G_s = I - R^{1/2} S^{-1/2} R^{1/2}``, the exact-gradient shrinkage
    correction ``G_eff = (1 - rho) G_s + (rho / d) tr(G_s) I`` (N.3), and the
    lower clamp on the spectrum of ``G_eff``.  ``ref_raw`` is only used to
    report the reference variance along the most negative direction.
    """
    d = sigma_g_tilde.shape[0]
    device, dtype = sigma_g_tilde.device, torch.float64
    identity = torch.eye(d, device=device, dtype=dtype)

    S = _symmetrize(ref_tilde_half @ sigma_g_tilde @ ref_tilde_half)
    lam, U = torch.linalg.eigh(S)
    lam_min, lam_max = lam.min(), lam.max()
    cond_S = (lam_max / lam_min).item() if lam_min.item() > 0.0 else math.inf
    # Not a floor: only a guard against 1/0 if a covariance is exactly singular,
    # which shrinkage rules out in normal operation.
    sqrt_lam = lam.clamp_min(torch.finfo(dtype).tiny).sqrt()
    S_inv_half = (U / sqrt_lam) @ U.T
    G_s = _symmetrize(identity - ref_tilde_half @ S_inv_half @ ref_tilde_half)
    G_eff = (1.0 - shrinkage) * G_s + (shrinkage / d) * torch.trace(G_s) * identity

    lam_G, U_G = torch.linalg.eigh(G_eff)
    min_index = int(torch.argmin(lam_G))
    G_min_eig = lam_G[min_index].item()
    u = U_G[:, min_index]
    G_min_eig_ref_var = (u @ ref_raw @ u).item()
    clamp_frac = 0.0
    if clamp is not None:
        below = lam_G < -clamp
        clamp_frac = below.double().mean().item()
        if bool(below.any()):
            lam_G = lam_G.clamp_min(-clamp)
            G_eff = _symmetrize((U_G * lam_G) @ U_G.T)

    fd_cov = (torch.trace(ref_tilde) + torch.trace(sigma_g_tilde) - 2.0 * sqrt_lam.sum()).item()
    return BuresGradient(G_eff, lam_G, U_G, fd_cov, cond_S, G_min_eig, G_min_eig_ref_var, clamp_frac)


class FdMoments:
    """EMA of raw feature moments, checkpointed alongside the epoch state.

    Holds float64 ``mu``, ``M`` (pooled) and ``M_resid`` (within-prompt
    residual second moment, whose mean is zero by construction), the number of
    EMA updates since the warm start, the group size the residuals were built
    with, and the reference cache key the warm start came from.
    """

    def __init__(self):
        self.mu = None
        self.M = None
        self.M_resid = None
        self.age = 0
        self.group_size = None
        self.reference_key = None

    def is_initialized(self):
        return self.mu is not None

    def to(self, device):
        if self.is_initialized():
            self.mu = self.mu.to(device)
            self.M = self.M.to(device)
            self.M_resid = self.M_resid.to(device)
        return self

    def warm_start(self, features, group_size, reference_key=None):
        features = features.to(torch.float64)
        self.mu, self.M = _raw_moments(features)
        _, self.M_resid = _raw_moments(residual_features(features, group_size))
        self.age = 0
        self.group_size = int(group_size)
        self.reference_key = reference_key

    def update(self, features, group_size, beta):
        if not self.is_initialized():
            raise RuntimeError("FdMoments.update called before warm_start")
        if int(group_size) != self.group_size:
            raise ValueError(
                f"group size changed from {self.group_size} to {int(group_size)}; "
                "the residual moments are not comparable across group sizes"
            )
        beta = float(beta)
        self.to(features.device)
        features = features.to(torch.float64)
        mu_batch, M_batch = _raw_moments(features)
        _, M_resid_batch = _raw_moments(residual_features(features, group_size))
        self.mu = beta * self.mu + (1.0 - beta) * mu_batch
        self.M = _symmetrize(beta * self.M + (1.0 - beta) * M_batch)
        self.M_resid = _symmetrize(beta * self.M_resid + (1.0 - beta) * M_resid_batch)
        self.age += 1

    def covariance(self):
        return _symmetrize(self.M - torch.outer(self.mu, self.mu))

    def covariance_resid(self):
        return self.M_resid

    def state_dict(self):
        if not self.is_initialized():
            return {"initialized": False}
        return {
            "initialized": True,
            "mu": self.mu.detach().cpu(),
            "M": self.M.detach().cpu(),
            "M_resid": self.M_resid.detach().cpu(),
            "age": int(self.age),
            "group_size": int(self.group_size),
            "reference_key": self.reference_key,
        }

    def load_state_dict(self, state):
        if not state.get("initialized", False):
            self.__init__()
            return
        self.mu = state["mu"].to(torch.float64)
        self.M = state["M"].to(torch.float64)
        self.M_resid = state["M_resid"].to(torch.float64)
        self.age = int(state["age"])
        self.group_size = int(state["group_size"])
        self.reference_key = state.get("reference_key")


@dataclass
class FdReference:
    """Fixed base-model statistics: raw moments plus the shrunk covariances and their roots."""

    mu: torch.Tensor
    sigma: torch.Tensor
    sigma_resid: torch.Tensor
    sigma_tilde: torch.Tensor
    sigma_tilde_half: torch.Tensor
    sigma_resid_tilde: torch.Tensor
    sigma_resid_tilde_half: torch.Tensor
    group_size: int
    num_samples: int
    floor_frac: float
    floor_frac_resid: float
    cache_key: str = None

    _TENSOR_FIELDS = (
        "mu", "sigma", "sigma_resid", "sigma_tilde", "sigma_tilde_half",
        "sigma_resid_tilde", "sigma_resid_tilde_half",
    )

    def to(self, device):
        for name in self._TENSOR_FIELDS:
            setattr(self, name, getattr(self, name).to(device))
        return self

    @classmethod
    def from_features(cls, features, group_size, config, cache_key=None):
        features = features.to(torch.float64)
        shrinkage = float(config.fd_shrinkage)
        eig_floor = float(config.fd_eig_floor)
        mu, M = _raw_moments(features)
        sigma = _symmetrize(M - torch.outer(mu, mu))
        _, sigma_resid = _raw_moments(residual_features(features, group_size))
        sigma_tilde, sigma_tilde_half, _, floor_frac = prepare_covariance(sigma, shrinkage, eig_floor)
        sigma_resid_tilde, sigma_resid_tilde_half, _, floor_frac_resid = prepare_covariance(
            sigma_resid, shrinkage, eig_floor
        )
        return cls(
            mu=mu,
            sigma=sigma,
            sigma_resid=sigma_resid,
            sigma_tilde=sigma_tilde,
            sigma_tilde_half=sigma_tilde_half,
            sigma_resid_tilde=sigma_resid_tilde,
            sigma_resid_tilde_half=sigma_resid_tilde_half,
            group_size=int(group_size),
            num_samples=int(features.shape[0]),
            floor_frac=floor_frac,
            floor_frac_resid=floor_frac_resid,
            cache_key=cache_key,
        )


@dataclass
class FdRewardOutput:
    reward: torch.Tensor  # (n,) float64, before group centering
    mean_term: torch.Tensor  # (n,) 2 (mu_g - mu_r)^T f_p, before weighting
    cov_term: torch.Tensor  # (n,) quadratic form of the configured source, before weighting
    fd: float
    fd_mean: float
    fd_cov: float
    fd_resid: float
    diagnostics: dict


def compute_fd_reward(features, group_size, moments, reference, config):
    """Per-sample FD reward and the epoch's FD values from the *previous* EMA state.

    Implements Sections 3.2 to 3.5 of the plan.  ``features`` are the epoch's
    gathered features ``(n, d)`` in contiguous prompt groups; the EMA update is
    the caller's job and happens after this call.
    """
    if not moments.is_initialized():
        raise RuntimeError("FdMoments must be warm-started before computing rewards")
    group_size = int(group_size)
    device = features.device
    f = features.to(torch.float64)
    n, d = f.shape
    if n % group_size != 0:
        raise ValueError(f"number of samples ({n}) must be divisible by the group size ({group_size})")
    moments.to(device)
    reference.to(device)

    shrinkage = float(config.fd_shrinkage)
    eig_floor = float(config.fd_eig_floor)
    clamp = None if config.fd_clamp is None else float(config.fd_clamp)
    w_mean = float(config.fd_mean_weight)
    w_cov = float(config.fd_cov_weight)
    source = config.fd_cov_source
    if source not in FD_COV_SOURCES:
        raise ValueError(f"fd_cov_source must be one of {FD_COV_SOURCES}, got {source!r}")

    mu_g = moments.mu
    sigma_g_tilde, _, lam_g_raw, floor_frac_pooled = prepare_covariance(
        moments.covariance(), shrinkage, eig_floor
    )
    pooled = bures_gradient(
        sigma_g_tilde, reference.sigma_tilde, reference.sigma_tilde_half, reference.sigma,
        shrinkage, clamp,
    )
    sigma_gr_tilde, _, lam_gr_raw, floor_frac_resid = prepare_covariance(
        moments.covariance_resid(), shrinkage, eig_floor
    )
    resid = bures_gradient(
        sigma_gr_tilde, reference.sigma_resid_tilde, reference.sigma_resid_tilde_half,
        reference.sigma_resid, shrinkage, clamp,
    )

    drift = mu_g - reference.mu
    fd_mean = drift.dot(drift).item()
    mean_term = 2.0 * (f @ drift)

    if source == "pooled":
        centered = f - mu_g
        cov_term = ((centered @ pooled.G) * centered).sum(dim=1)
        active, active_features = pooled, centered
        floor_frac, lam_raw = floor_frac_pooled, lam_g_raw
    else:
        # Section 3.6: the residual objective sees each sample through its
        # group mean too, which adds (1/N) times the uncentered quadratic form.
        resid_f = residual_features(f, group_size)
        cov_term = ((resid_f @ resid.G) * resid_f).sum(dim=1) + ((f @ resid.G) * f).sum(dim=1) / group_size
        active, active_features = resid, resid_f
        floor_frac, lam_raw = floor_frac_resid, lam_gr_raw

    reward = -(w_mean * mean_term + w_cov * cov_term)

    # Fraction of the covariance term's energy carried by the top eigen-directions of G.
    energy = active.eigvals.abs() * ((active_features @ active.eigvecs) ** 2).sum(dim=0)
    top_k = energy.topk(min(8, d)).values.sum()
    energy_top8 = (top_k / energy.sum().clamp_min(torch.finfo(torch.float64).tiny)).item()

    diagnostics = {
        "fd_cond_S": active.cond_S,
        "fd_neg_eig_sigma_g": int((lam_raw < 0).sum()),
        "fd_G_min_eig": active.G_min_eig,
        "fd_G_min_eig_ref_var": active.G_min_eig_ref_var,
        "fd_clamp_frac": active.clamp_frac,
        "fd_floor_frac": floor_frac,
        "fd_cov_energy_top8": energy_top8,
        "fd_ema_age": float(moments.age),
    }
    return FdRewardOutput(
        reward=reward,
        mean_term=mean_term,
        cov_term=cov_term,
        fd=fd_mean + pooled.fd_cov,
        fd_mean=fd_mean,
        fd_cov=pooled.fd_cov,
        fd_resid=resid.fd_cov,
        diagnostics=diagnostics,
    )


def clip_combined_reward(combined, group_size, clip):
    """Clip the combined FD reward at ``clip`` standard deviations, then re-center.

    ``clip`` is ``fd_sample_clip``.  The limit is ``clip * std(combined)`` with
    the population std of the *pre-clip* vector; the clipped vector is centered
    per prompt group again, because the clip is not group-symmetric and would
    otherwise leave a group mean behind.  Bounds the per-sample outliers the
    covariance term produces once its matrix turns indefinite.

    Returns ``(clipped, clip_frac)``, the fraction being the share of samples
    that hit the limit.  ``clip=None`` is a no-op returning the input object, so
    a config without the key behaves exactly as before the key existed.
    """
    if clip is None:
        return combined, 0.0
    limit = float(clip) * combined.std(correction=0).item()
    clip_frac = (combined.abs() > limit).double().mean().item()
    return center_by_group(combined.clamp(min=-limit, max=limit), int(group_size)), clip_frac


def capped_fd_scale(fd_scale, spread, realized_max):
    """``fd_scale`` after the per-epoch cap on the realized strength.

    ``spread`` is ``std(combined) / pre_fd_std``, so ``scale * spread`` is the
    dose the FD term adds relative to the pre-FD advantage spread.
    ``realized_max`` is ``fd_realized_max``: the scale is lowered to
    ``realized_max / spread`` when the dose would exceed it, and never raised
    (one-sided, so the near-zero signal of the EMA warm-in is not amplified).
    At zero spread -- the warm-started EMA gives an exactly zero reward in the
    first epochs -- the scale is zero rather than a division by zero; the delta
    is zero either way.
    """
    scale = float(fd_scale)
    if realized_max is None:
        return scale
    if spread > 0.0:
        return min(scale, float(realized_max) / float(spread))
    return 0.0


# ---------------------------------------------------------------------------
# Spaces and the per-epoch driver
# ---------------------------------------------------------------------------


@dataclass
class FdSpace:
    """One representation space: its encoder, reference, EMA and role."""

    name: str
    encoder: object  # anything with ``embed(images) -> (n, d)``
    reference: FdReference
    moments: FdMoments
    null_fd: float
    trained: bool
    weight: float


class FdSpaces:
    """Every loaded space, and the single object registered for checkpointing.

    Built before ``load_state`` with empty moments for every configured space
    (trained first, then log-only), so a resumed checkpoint restores the EMA of
    each.  The saved space set must equal the configured one and the saved
    reference key must match the config's; otherwise loading raises.
    ``attach`` supplies the encoder, reference and null-FD once they exist.
    """

    def __init__(self, config, reference_key):
        self.reference_key = reference_key
        self.spaces = {}
        for name in fd_loaded_types(config):
            trained = name in config.fd_train_types
            weight = config.fd_train_weights[config.fd_train_types.index(name)] if trained else 0.0
            self.spaces[name] = FdSpace(
                name=name, encoder=None, reference=None, moments=FdMoments(), null_fd=None,
                trained=trained, weight=float(weight),
            )

    def __iter__(self):
        return iter(self.spaces.values())

    def __len__(self):
        return len(self.spaces)

    def __getitem__(self, name):
        return self.spaces[name]

    @property
    def names(self):
        return list(self.spaces)

    @property
    def trained_names(self):
        return [name for name, space in self.spaces.items() if space.trained]

    def attach(self, name, encoder, reference, null_fd):
        space = self.spaces[name]
        space.encoder, space.reference, space.null_fd = encoder, reference, float(null_fd)

    def is_initialized(self):
        return all(space.moments.is_initialized() for space in self)

    @property
    def age(self):
        return self.spaces[next(iter(self.spaces))].moments.age

    def state_dict(self):
        return {
            "reference_key": self.reference_key,
            "moments": {name: space.moments.state_dict() for name, space in self.spaces.items()},
        }

    def load_state_dict(self, state):
        saved = sorted(state.get("moments", {}))
        if saved != sorted(self.spaces):
            raise RuntimeError(
                f"Resumed FD checkpoint holds spaces {saved}, but the config loads {sorted(self.spaces)}; "
                "keep fd_train_types and fd_log_types fixed within one output_dir"
            )
        if state.get("reference_key") != self.reference_key:
            raise RuntimeError(
                f"Resumed FD moments were warm-started from reference {state.get('reference_key')}, "
                f"but the config resolves to reference {self.reference_key}"
            )
        for name, space in self.spaces.items():
            space.moments.load_state_dict(state["moments"][name])


@dataclass
class FdEpochResult:
    """What ``sample_epoch`` needs back from one active epoch."""

    advantage_delta: torch.Tensor  # local (n_local,) term to add, or None at scale zero
    tensors: dict  # per-sample local tensors, reduced to means by the caller
    scalars: dict  # identical on every rank
    combined: torch.Tensor  # the gathered (n,) float64 combined reward at unit scale


class FdTerm:
    """The FD-to-base term as ``sample_epoch`` sees it: embed, reward, combine, EMA update.

    Per loaded space, in a fixed order: the epoch's images are embedded and
    all-gathered so the moments and the reward use every sample, the reward is
    taken against the previous EMA state, centered per prompt group, and logged
    under ``<key>/<space>``.  Trained spaces contribute ``weight / denominator``
    times their centered reward to the combined vector;
    the unsuffixed scalars describe that vector.  Every EMA is updated after
    all spaces have computed their reward.
    """

    def __init__(self, spaces, config):
        self.spaces = spaces
        self.config = config

    def denominator(self, space, fd_value):
        """``D_i`` of the configured ``fd_normalize`` mode for a trained space."""
        mode = self.config.fd_normalize
        if mode == "none":
            return 1.0
        if mode == "null_ratio":
            return float(space.null_fd)
        if mode == "sg_ratio":
            eps = self.config.fd_norm_eps
            return float(fd_value) + (float(space.null_fd) if eps is None else float(eps))
        raise ValueError(f"fd_normalize must be one of {FD_NORMALIZE_MODES}, got {mode!r}")

    def compute_epoch(self, images, group_size, advantages, accelerator, epoch):
        """Return an ``FdEpochResult``, or ``None`` when ``epoch`` is inactive.

        ``advantages`` is the local pre-FD advantage; it is only read, for the
        spread ratios.  ``group_size`` is ``num_samples_per_prompt``.
        """
        config = self.config
        if not fd_is_active_epoch(config, epoch):
            return None
        G = int(group_size)
        tensors = {}
        scalars = {}
        # Calibration quantities: spreads of the centered FD rewards against
        # the spread of the advantage they would be added to (task reward, plus
        # the DreamSim term when that is on).  Meaningful at fd_scale 0 too.
        pre_fd_std = accelerator.gather(advantages).double().std(correction=0) + 1e-12

        rank = accelerator.process_index
        local = None
        combined = None
        ratios = []
        pending_updates = []
        for space in self.spaces:
            name = space.name
            features = space.encoder.embed(images)
            if name == "inception_v3":
                tensors["vendi_score/inception_v3"] = grouped_vendi_scores(features, G)
            features_all = accelerator.gather(features)
            n_local = features.shape[0]
            local = slice(rank * n_local, (rank + 1) * n_local)
            assert torch.equal(features_all[local], features), f"gathered {name} features are not rank-major"

            out = compute_fd_reward(features_all, G, space.moments, space.reference, config)
            centered = center_by_group(out.reward, G)
            scalars[f"fd/{name}"] = out.fd
            scalars[f"fd_mean/{name}"] = out.fd_mean
            scalars[f"fd_cov/{name}"] = out.fd_cov
            scalars[f"fd_resid/{name}"] = out.fd_resid
            scalars[f"fd_null/{name}"] = space.null_fd
            scalars[f"fd_ratio/{name}"] = out.fd / space.null_fd
            ratios.append(out.fd / space.null_fd)
            scalars[f"fd_reward_spread_ratio/{name}"] = (centered.std(correction=0) / pre_fd_std).item()
            for key, value in out.diagnostics.items():
                if key != "fd_ema_age":
                    scalars[f"{key}/{name}"] = value
            tensors[f"fd_mean_term_abs/{name}"] = out.mean_term[local].abs()
            tensors[f"fd_cov_term_abs/{name}"] = out.cov_term[local].abs()
            if space.trained:
                denominator = self.denominator(space, out.fd)
                scalars[f"fd_norm/{name}"] = denominator
                contribution = (space.weight / denominator) * centered
                combined = contribution if combined is None else combined + contribution
            pending_updates.append((space, features_all))

        scalars["fd_ema_age"] = float(self.spaces.age)
        scalars["fd_ratio_mean"] = sum(ratios) / len(ratios)
        # The clip reshapes the vector everything downstream sees: the spread
        # ratio, the cap, the delta and the returned combined reward.  The
        # per-space ratios above are of the uncombined, unclipped rewards.
        combined, clip_frac = clip_combined_reward(
            combined, G, getattr(config, "fd_sample_clip", None)
        )
        scalars["fd_clip_frac"] = clip_frac
        spread = (combined.std(correction=0) / pre_fd_std).item()
        scalars["fd_reward_spread_ratio"] = spread
        # The eigensolves are float64 on identical inputs, so ranks should agree
        # to the bit; any drift shows up here rather than silently desyncing.
        rank0_combined = broadcast(combined.clone(), from_process=0)
        scalars["fd_rank_max_diff"] = reduce_max(accelerator, (combined - rank0_combined).abs().max())

        scale = capped_fd_scale(config.fd_scale, spread, getattr(config, "fd_realized_max", None))
        scalars["fd_scale_effective"] = scale
        # The dose actually applied to the advantage: the spread the FD term adds
        # relative to the pre-FD advantage spread.  Defined (as 0) at fd_scale 0.
        scalars["fd_realized_strength"] = scale * spread

        advantage_delta = None
        if fd_reward_is_enabled(config):
            advantage_delta = scale * combined[local].to(
                device=advantages.device, dtype=advantages.dtype
            )
            tensors["fd_advantage_abs"] = advantage_delta.abs()
        for space, features_all in pending_updates:
            space.moments.update(features_all, G, config.fd_ema_beta)
        return FdEpochResult(advantage_delta=advantage_delta, tensors=tensors, scalars=scalars, combined=combined)


__all__ = [
    "FD_COV_SOURCES", "FD_NORMALIZE_MODES", "FdEpochResult", "FdMoments", "FdReference", "FdRewardOutput",
    "FdSpace", "FdSpaces", "FdTerm", "apply_fd_config_defaults", "bures_gradient", "capped_fd_scale",
    "center_by_group", "clip_combined_reward", "compute_fd_reward", "fd_is_active_epoch",
    "fd_is_enabled", "fd_loaded_types", "fd_logging_is_enabled", "fd_reward_is_enabled",
    "prepare_covariance", "residual_features",
]
