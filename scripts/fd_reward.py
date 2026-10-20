"""Fréchet distance to the base model as a policy-gradient reward.

The Fréchet distance (FD) between the fine-tuned generator and the base model,
measured in a frozen encoder's feature space, depends on the policy only
through the raw feature moments ``mu_g = E[f]`` and ``M_g = E[f f^T]``.  The
score-function identity therefore turns its gradient into an ordinary policy
gradient whose per-sample reward is a scalar quadratic form in the feature::

    r_p = -[2 (mu_g - mu_r)^T f_p + u_p^T G u_p + f_p^T G f_p / N]
    u_p = f_p - mean(f within the prompt group)
    G   = I - Sigma_r^{1/2} (Sigma_r^{1/2} Sigma_g Sigma_r^{1/2})^{-1/2} Sigma_r^{1/2}

The covariance matrices in ``G`` are within-prompt residual covariances. The
``1/N`` correction accounts for each sample's contribution to its group mean.
The moments are held in a per-epoch EMA warm-started from base-model features.
No gradients flow through the encoder. The numerics are float64, with
shrinkage (and its exact-gradient correction) and a clamp on the spectrum of
``G``. Each space's reward is divided by its base-model null-FD before mixing.

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
        "fd_log_types": [],
        "fd_scale": 0.0,
        "fd_ema_beta": 0.95,
        "fd_init_epochs": 25,
        "fd_shrinkage": 0.02,
        "fd_clamp": 10.0,
        "fd_eig_floor": 0.0,
        # Diagnostic spaces follow this interval; training spaces run every epoch.
        "fd_log_every_n_epochs": 0,
        "fd_cache_dir": "models/fd_stats",
        "fd_encoder_cache_dir": "models/fd_encoders",
        "fd_batch_size": None,
        "fd_reference_seed": None,
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
    if config.fd_scale is None:
        config.fd_scale = 0.0
    config.fd_scale = _finite_float(config.fd_scale, "fd_scale")
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

    if fd_is_enabled(config):
        if config.num_samples_per_prompt < 2:
            raise ValueError(
                "Residual FD covariance needs num_samples_per_prompt >= 2, got "
                f"{config.num_samples_per_prompt}"
            )
        reference_samples = (
            config.fd_init_epochs * config.num_prompts_per_epoch * config.num_samples_per_prompt
        )
        widest = max(fd_loaded_types(config), key=lambda name: resolve_fd_type(name).dim)
        dim = resolve_fd_type(widest).dim
        if reference_samples < 2 * dim:
            raise ValueError(
                "The FD reference pass needs at least 2 x d samples: fd_init_epochs x "
                "num_prompts_per_epoch x num_samples_per_prompt = "
                f"{reference_samples} < 2 x {dim} for {widest}. "
                "Raise fd_init_epochs."
            )
    return config


def fd_loaded_types(config):
    """Active training spaces, followed by enabled diagnostic spaces."""
    names = list(config.fd_train_types) if fd_reward_is_enabled(config) else []
    for name in config.fd_log_types if fd_logging_is_enabled(config) else []:
        if name not in names:
            names.append(name)
    return names


def fd_reward_is_enabled(config):
    """True when the FD term is added to the advantages."""
    return float(config.fd_scale) != 0.0


def fd_logging_is_enabled(config):
    """True when FD is measured during ``sample_epoch`` without entering the advantage."""
    return bool(config.fd_log_types) and config.fd_log_every_n_epochs > 0


def fd_is_enabled(config):
    """True when the encoder, the reference pass and the EMA are needed at all."""
    return fd_reward_is_enabled(config) or fd_logging_is_enabled(config)


def fd_is_active_epoch(config, epoch):
    """True when the FD term is computed in ``sample_epoch`` for ``epoch``.

    Training spaces run every epoch when the reward is enabled. Diagnostic
    spaces follow ``fd_log_every_n_epochs``; their EMA updates only when measured.
    """
    if fd_reward_is_enabled(config):
        return True
    interval = config.fd_log_every_n_epochs
    return fd_logging_is_enabled(config) and epoch % interval == 0


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

    Uses ``S = R^{1/2} Sigma_g R^{1/2}``,
    ``G_s = I - R^{1/2} S^{-1/2} R^{1/2}``, the exact-gradient shrinkage
    correction ``G_eff = (1 - rho) G_s + (rho / d) tr(G_s) I``, and the
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
    mean_term: torch.Tensor  # (n,) 2 (mu_g - mu_r)^T f_p
    cov_term: torch.Tensor  # (n,) residual covariance contribution, including the 1/N correction
    fd: float
    fd_mean: float
    fd_cov: float
    fd_resid: float
    diagnostics: dict


def compute_fd_reward(features, group_size, moments, reference, config):
    """Per-sample FD reward and the epoch's FD values from the *previous* EMA state.

    ``features`` are the epoch's gathered features ``(n, d)`` in contiguous
    prompt groups; the caller updates the EMA after this call. Each measured
    space reports both pooled FD and residual covariance distance.
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
    mu_g = moments.mu
    sigma_g_tilde, _, _, _ = prepare_covariance(
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

    # The residual objective also depends on each sample through its group
    # mean, adding the uncentered quadratic form divided by the group size.
    resid_f = residual_features(f, group_size)
    cov_term = ((resid_f @ resid.G) * resid_f).sum(dim=1) + ((f @ resid.G) * f).sum(dim=1) / group_size
    active, active_features = resid, resid_f
    floor_frac, lam_raw = floor_frac_resid, lam_gr_raw

    reward = -(mean_term + cov_term)

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
            trained = fd_reward_is_enabled(config) and name in config.fd_train_types
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
    combined: torch.Tensor | None  # gathered reward at unit scale, or None for diagnostics only


class FdTerm:
    """The FD-to-base term as ``sample_epoch`` sees it: embed, reward, combine, EMA update.

    Per loaded space, in a fixed order: the epoch's images are embedded and
    all-gathered so the moments and the reward use every sample, the reward is
    taken against the previous EMA state, centered per prompt group, and logged
    under ``<key>/<space>``. Trained spaces contribute ``weight / null_fd``
    times their centered reward to the combined vector;
    the unsuffixed scalars describe that vector.  Every EMA is updated after
    all spaces have computed their reward.
    """

    def __init__(self, spaces, config):
        self.spaces = spaces
        self.config = config

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
            if not space.trained and epoch % config.fd_log_every_n_epochs != 0:
                continue
            name = space.name
            features = space.encoder.embed(images)
            features_all = accelerator.gather(features)
            n_local = features.shape[0]
            local = slice(rank * n_local, (rank + 1) * n_local)
            assert torch.equal(features_all[local], features), f"gathered {name} features are not rank-major"

            out = compute_fd_reward(features_all, G, space.moments, space.reference, config)
            centered = center_by_group(out.reward, G)
            scalars[f"fd_mean/{name}"] = out.fd_mean
            scalars[f"fd_resid/{name}"] = out.fd_resid
            scalars[f"fd_null/{name}"] = space.null_fd
            scalars[f"fd/{name}"] = out.fd
            scalars[f"fd_cov/{name}"] = out.fd_cov
            scalars[f"fd_ratio/{name}"] = out.fd / space.null_fd
            ratios.append(out.fd / space.null_fd)
            scalars[f"fd_reward_spread_ratio/{name}"] = (centered.std(correction=0) / pre_fd_std).item()
            for key, value in out.diagnostics.items():
                if key != "fd_ema_age":
                    scalars[f"{key}/{name}"] = value
            tensors[f"fd_mean_term_abs/{name}"] = out.mean_term[local].abs()
            tensors[f"fd_cov_term_abs/{name}"] = out.cov_term[local].abs()
            if space.trained:
                denominator = float(space.null_fd)
                scalars[f"fd_norm/{name}"] = denominator
                contribution = (space.weight / denominator) * centered
                combined = contribution if combined is None else combined + contribution
            pending_updates.append((space, features_all))

        scalars["fd_ema_age"] = float(self.spaces.age)
        if ratios:
            scalars["fd_ratio_mean"] = sum(ratios) / len(ratios)
        advantage_delta = None
        if combined is not None:
            spread = (combined.std(correction=0) / pre_fd_std).item()
            scalars["fd_reward_spread_ratio"] = spread
            # Identical inputs should give identical rewards across ranks.
            rank0_combined = broadcast(combined.clone(), from_process=0)
            scalars["fd_rank_max_diff"] = reduce_max(accelerator, (combined - rank0_combined).abs().max())
            scale = float(config.fd_scale)
            scalars["fd_scale_effective"] = scale
            scalars["fd_realized_strength"] = scale * spread
            advantage_delta = scale * combined[local].to(
                device=advantages.device, dtype=advantages.dtype
            )
            tensors["fd_advantage_abs"] = advantage_delta.abs()
        for space, features_all in pending_updates:
            space.moments.update(features_all, G, config.fd_ema_beta)
        return FdEpochResult(advantage_delta=advantage_delta, tensors=tensors, scalars=scalars, combined=combined)


__all__ = [
    "FdEpochResult", "FdMoments", "FdReference", "FdRewardOutput",
    "FdSpace", "FdSpaces", "FdTerm", "apply_fd_config_defaults", "bures_gradient",
    "center_by_group", "compute_fd_reward", "fd_is_active_epoch",
    "fd_is_enabled", "fd_loaded_types", "fd_logging_is_enabled", "fd_reward_is_enabled",
    "prepare_covariance", "residual_features",
]
