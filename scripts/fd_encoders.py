"""Frozen representation encoders for the FD-to-base term.

The registry names the six representation models of the FD-Loss paper (Yang et
al., arXiv:2604.28190) and ``FdEncoder`` embeds ``[0, 1]`` image tensors with
one of them.  Loading goes through ``transformers`` / ``torchvision`` (the lock
pins timm 0.6.13, which predates DINOv2), so numbers are not bit-identical to
the paper's timm ports.  Features are raw, never L2-normalized; the math that
consumes them lives in ``fd_reward.py`` and knows nothing about encoders.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Encoder registry
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class FdEncoderSpec:
    dim: int
    image_size: int
    mean: tuple
    std: tuple
    source: str


# The six representation models of the FD-Loss paper, loaded through
# ``transformers`` / ``torchvision``.  ``source`` documents the exact feature.
FD_ENCODERS = {
    "dinov2_vitl14": FdEncoderSpec(
        1024, 224, IMAGENET_MEAN, IMAGENET_STD,
        "transformers Dinov2Model facebook/dinov2-large; CLS token after the final layernorm",
    ),
    "inception_v3": FdEncoderSpec(
        2048, 299, IMAGENET_MEAN, IMAGENET_STD,
        "torchvision inception_v3 IMAGENET1K_V1 with fc = Identity; global average pool",
    ),
    "convnextv2_base": FdEncoderSpec(
        1024, 224, IMAGENET_MEAN, IMAGENET_STD,
        "transformers ConvNextV2Model facebook/convnextv2-base-22k-224 (FCMAE, IN-22k then IN-1k); "
        "global average pool of the last hidden state, before the head layernorm",
    ),
    "mae_vitl16": FdEncoderSpec(
        1024, 224, IMAGENET_MEAN, IMAGENET_STD,
        "transformers ViTMAEModel facebook/vit-mae-large with mask_ratio=0 and identity noise; "
        "CLS token after the final layernorm",
    ),
    "siglip2_so400m": FdEncoderSpec(
        1152, 256, SIGLIP_MEAN, SIGLIP_STD,
        "transformers SiglipVisionModel google/siglip2-so400m-patch16-256; attention-pooled output",
    ),
    "clip_vitl14": FdEncoderSpec(
        1024, 224, CLIP_MEAN, CLIP_STD,
        "transformers CLIPVisionModel openai/clip-vit-large-patch14; CLS after post-layernorm, before projection",
    ),
}
DEFAULT_FD_TYPE = "dinov2_vitl14"


def resolve_fd_type(name):
    """Map a config encoder name to its ``FdEncoderSpec``."""
    if not isinstance(name, str) or name not in FD_ENCODERS:
        raise ValueError(f"Unknown FD encoder {name!r}; choose from {sorted(FD_ENCODERS)}")
    return FD_ENCODERS[name]


# ---------------------------------------------------------------------------
# Loaders and feature functions, one pair per registry name
# ---------------------------------------------------------------------------
#
# Every loader returns a CPU module from the offline cache under ``cache_dir``;
# every feature function maps a preprocessed ``(B, 3, S, S)`` batch to the raw
# ``(B, d)`` feature of the spec.  ``FdEncoder`` owns device placement, eval
# mode, autocast and chunking.


def _load_dinov2(cache_dir):
    from transformers import Dinov2Model

    return Dinov2Model.from_pretrained("facebook/dinov2-large", cache_dir=str(cache_dir))


def _features_dinov2(model, batch):
    # pooler_output is the CLS token after the final layernorm.
    return model(pixel_values=batch).pooler_output


def _load_inception(cache_dir):
    from torchvision.models import Inception_V3_Weights, inception_v3

    # Built the way torchvision builds it with pretrained weights (aux head so
    # the state dict loads strictly, transform_input=True), but with the weight
    # file under our cache directory rather than TORCH_HOME.
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=None, aux_logits=True, transform_input=True, init_weights=False)
    model.load_state_dict(
        weights.get_state_dict(model_dir=str(Path(cache_dir) / "torchvision"), check_hash=True)
    )
    model.fc = nn.Identity()
    return model


def _features_inception(model, batch):
    # transform_input remaps ImageNet-normalized input to the [-1, 1] scaling
    # the weights were trained with; in eval mode the module returns the
    # 2048-d pooled tensor (fc is the identity), not InceptionOutputs.
    return model(batch)


def _load_convnextv2(cache_dir):
    from transformers import ConvNextV2Model

    return ConvNextV2Model.from_pretrained("facebook/convnextv2-base-22k-224", cache_dir=str(cache_dir))


def _features_convnextv2(model, batch):
    # Global average pool of the (B, C, H, W) last hidden state.  Not
    # pooler_output, which applies the head layernorm after pooling; FD-Loss
    # takes the pre-norm pooled feature.
    return model(pixel_values=batch).last_hidden_state.mean(dim=(2, 3))


def _load_mae(cache_dir):
    from transformers import ViTMAEModel

    return ViTMAEModel.from_pretrained("facebook/vit-mae-large", cache_dir=str(cache_dir), mask_ratio=0.0)


def _features_mae(model, batch):
    # With mask_ratio 0 every patch is kept, and an explicit arange noise makes
    # the argsort shuffle the identity without touching the CUDA RNG stream.
    num_patches = model.embeddings.num_patches
    noise = torch.arange(num_patches, device=batch.device, dtype=torch.float32).expand(batch.shape[0], -1)
    return model(pixel_values=batch, noise=noise).last_hidden_state[:, 0]


def _load_siglip2(cache_dir):
    from transformers import SiglipVisionModel

    # The so400m-patch16-256 checkpoint declares model_type "siglip" (fixed
    # 256 px grid), so the plain SigLIP vision tower loads it.
    return SiglipVisionModel.from_pretrained("google/siglip2-so400m-patch16-256", cache_dir=str(cache_dir))


def _features_siglip2(model, batch):
    # pooler_output is the attention-pooling head over the post-layernorm tokens.
    return model(pixel_values=batch).pooler_output


def _load_clip(cache_dir):
    from transformers import CLIPVisionModel

    return CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=str(cache_dir))


def _features_clip(model, batch):
    # pooler_output is the CLS token after post_layernorm, before the
    # projection that CLIPModel applies; QuickGELU comes from the config.
    return model(pixel_values=batch).pooler_output


_IMPLEMENTATIONS = {
    "dinov2_vitl14": (_load_dinov2, _features_dinov2),
    "inception_v3": (_load_inception, _features_inception),
    "convnextv2_base": (_load_convnextv2, _features_convnextv2),
    "mae_vitl16": (_load_mae, _features_mae),
    "siglip2_so400m": (_load_siglip2, _features_siglip2),
    "clip_vitl14": (_load_clip, _features_clip),
}
assert set(_IMPLEMENTATIONS) == set(FD_ENCODERS)


# ---------------------------------------------------------------------------
# Encoder module
# ---------------------------------------------------------------------------


class FdEncoder(nn.Module):
    """Embeds ``[0, 1]`` image tensors with one frozen representation model."""

    def __init__(self, type_name, cache_dir, batch_size, device):
        super().__init__()
        self.spec = resolve_fd_type(type_name)
        self.type_name = type_name
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        loader, self._feature_fn = _IMPLEMENTATIONS[type_name]
        self.model = loader(cache_dir).to(self.device).eval().requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor(self.spec.mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(self.spec.std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        )

    @property
    def dim(self):
        return self.spec.dim

    def _preprocess(self, images):
        # Same on-device bicubic resize as DreamSimScorer._resize, then the
        # encoder's mean/std normalization.  No L2 normalization of features.
        size = self.spec.image_size
        images = images.to(self.device, dtype=torch.float32)
        if images.shape[-2:] != (size, size):
            images = nn.functional.interpolate(
                images, size=(size, size), mode="bicubic", align_corners=False, antialias=True
            ).clamp(0.0, 1.0)
        return (images - self.mean) / self.std

    def _features(self, batch):
        return self._feature_fn(self.model, batch)

    @torch.no_grad()
    def embed(self, images):
        """``(B, 3, H, W)`` floats in ``[0, 1]`` -> ``(B, d)`` float32 raw features."""
        chunks = []
        autocast = torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
        )
        for start in range(0, images.shape[0], self.batch_size):
            batch = self._preprocess(images[start:start + self.batch_size])
            with autocast:
                features = self._features(batch)
            chunks.append(features.float())
        if not chunks:
            return torch.zeros((0, self.spec.dim), dtype=torch.float32, device=self.device)
        return torch.cat(chunks)


def prefetch(cache_dir, names=None):
    """Download every named encoder (default: all six) into ``cache_dir`` on the CPU.

    Run once from a node with network access; compute nodes load from the
    cache offline.  Returns the names that were loaded.
    """
    names = list(names or FD_ENCODERS)
    for name in names:
        resolve_fd_type(name)
    for name in names:
        loader, _ = _IMPLEMENTATIONS[name]
        model = loader(cache_dir)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: ready ({params / 1e6:.0f}M parameters)", flush=True)
        del model
    return names


__all__ = [
    "DEFAULT_FD_TYPE", "FD_ENCODERS", "FdEncoder", "FdEncoderSpec", "prefetch",
    "resolve_fd_type",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prefetch the FD encoder weights into an offline cache directory."
    )
    parser.add_argument("cache_dir", nargs="?", default="models/fd_encoders")
    parser.add_argument("--spaces", nargs="*", default=None, help="registry names; default all six")
    args = parser.parse_args()
    prefetch(args.cache_dir, args.spaces)
    print("FD encoder cache ready")
