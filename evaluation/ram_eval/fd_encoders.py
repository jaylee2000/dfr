"""Frozen Inception-v3 and DINOv2 features for FD, without L2 normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .metrics_common import iter_batches, open_rgb


@dataclass(frozen=True)
class FdEncoderSpec:
    dim: int
    image_size: int
    source: str


FD_ENCODERS = {
    "inception_v3": FdEncoderSpec(2048, 299, "torchvision Inception-v3 IMAGENET1K_V1 pooled"),
    "dinov2_vitl14": FdEncoderSpec(1024, 224, "facebook/dinov2-large CLS post-layernorm"),
}


class FdEncoder:
    def __init__(self, name: str, weights_path: Path, device: str):
        import torch

        self.name = name
        self.spec = FD_ENCODERS[name]
        self.device = torch.device(device)
        if name == "inception_v3":
            from torchvision.models import inception_v3

            self.model = inception_v3(weights=None, aux_logits=True, transform_input=True, init_weights=False)
            self.model.load_state_dict(
                torch.load(weights_path / "inception_v3_google-0cc3c7bd.pth", map_location="cpu", weights_only=True)
            )
            self.model.fc = torch.nn.Identity()
        else:
            from transformers import Dinov2Model

            self.model = Dinov2Model.from_pretrained(str(weights_path), local_files_only=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def extract(self, rows: list[dict], batch_size: int):
        import numpy as np
        import torch
        from torch.nn import functional as F
        from torchvision.transforms.functional import pil_to_tensor

        chunks = []
        for batch in iter_batches(rows, batch_size):
            tensors = []
            for row in batch:
                image = open_rgb(row["image_path"])
                try:
                    tensor = pil_to_tensor(image).float().div_(255.0).unsqueeze(0)
                finally:
                    image.close()
                # Match the evaluation feature spaces: resize directly, without a crop.
                if tensor.shape[-2:] != (self.spec.image_size, self.spec.image_size):
                    tensor = F.interpolate(
                        tensor, size=(self.spec.image_size,) * 2, mode="bicubic", align_corners=False, antialias=True
                    ).clamp_(0, 1)
                tensors.append(tensor.squeeze(0))
            images = torch.stack(tensors).to(self.device)
            images = (images - self.mean) / self.std
            with (
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"),
            ):
                if self.name == "inception_v3":
                    output = self.model(images)
                else:
                    output = self.model(pixel_values=images).pooler_output
            chunks.append(output.detach().cpu().float().numpy())
        return np.concatenate(chunks) if chunks else np.empty((0, self.spec.dim), dtype=np.float32)
