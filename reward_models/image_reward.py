"""ImageReward scorer (Xu et al., 2023). Wraps the BLIP-based regression head from
the upstream `ImageReward` package."""

import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODELS_ROOT = os.path.join(_REPO_ROOT, "models")
_HF_HOME = os.path.join(_MODELS_ROOT, ".huggingface")
os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_HOME, "hub"))
os.environ.setdefault("HF_XET_CACHE", os.path.join(_HF_HOME, "xet"))
os.makedirs(_MODELS_ROOT, exist_ok=True)
os.makedirs(_HF_HOME, exist_ok=True)

import transformers.modeling_utils as _transformers_modeling_utils
from transformers.pytorch_utils import (
    apply_chunking_to_forward as _apply_chunking_to_forward,
    find_pruneable_heads_and_indices as _find_pruneable_heads_and_indices,
    prune_linear_layer as _prune_linear_layer,
)

_TRANSFORMERS_MODELING_UTILS_COMPAT = {
    "apply_chunking_to_forward": _apply_chunking_to_forward,
    "find_pruneable_heads_and_indices": _find_pruneable_heads_and_indices,
    "prune_linear_layer": _prune_linear_layer,
}
for _name, _value in _TRANSFORMERS_MODELING_UTILS_COMPAT.items():
    if not hasattr(_transformers_modeling_utils, _name):
        setattr(_transformers_modeling_utils, _name, _value)

import ImageReward as RM
import torch
import torch.nn as nn
from ImageReward.models.BLIP.blip_pretrain import BLIP_Pretrain
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def _transform_tensor():
    return Compose(
        [
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

        # initial MLP param
        for name, param in self.layers.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
            if "bias" in name:
                nn.init.constant_(param, val=0)

    def forward(self, input):
        return self.layers(input)


class IRSMC(nn.Module):
    def __init__(self, med_config):
        super().__init__()

        self.blip = BLIP_Pretrain(image_size=224, vit="large", med_config=med_config)
        self.preprocess = _transform(224)
        self.preprocess_tensor = _transform_tensor()
        self.mlp = MLP(768)

        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

    @property
    def device(self):
        return next(self.parameters()).device

    def score(self, prompt, image):
        if type(image).__name__ == "list":
            _, rewards = self.inference_rank(prompt, image)
            return rewards

        # text encode
        text_input = self.blip.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str) and os.path.isfile(image):
            pil_image = Image.open(image)
        else:
            raise TypeError(
                r"This image parameter type has not been supportted yet. Please pass PIL.Image or file path str."
            )

        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_embeds = self.blip.visual_encoder(image)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards.detach().cpu().numpy().item()

    def score_batched(self, prompts, images):
        assert isinstance(prompts, list)
        assert isinstance(images, list)

        # text encode
        text_input = self.blip.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        # image encode
        images = [self.preprocess(image).unsqueeze(0).to(self.device) for image in images]
        images = torch.cat(images, 0).to(self.device)

        image_embeds = self.blip.visual_encoder(images)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards.view(txt_features.shape[0]).detach().cpu().numpy().tolist()

    def score_from_prompt_batched(self, prompts, images):
        assert isinstance(prompts, list)
        assert isinstance(images, list)

        # text encode
        text_input = self.blip.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        # image encode
        images = [self.preprocess_tensor(image).unsqueeze(0).to(self.device) for image in images]
        images = torch.cat(images, 0).to(self.device).float()
        image_embeds = self.blip.visual_encoder(images)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards.view(txt_features.shape[0])

    def inference_rank(self, prompt, generations_list):
        text_input = self.blip.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)
        txt_set = []
        for generation in generations_list:
            # image encode
            if isinstance(generation, Image.Image):
                pil_image = generation
            elif isinstance(generation, str):
                if os.path.isfile(generation):
                    pil_image = Image.open(generation)
            else:
                raise TypeError(
                    r"This image parameter type has not been supportted yet. Please pass PIL.Image or file path str."
                )
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_embeds = self.blip.visual_encoder(image)

            # text encode cross attention with image
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            text_output = self.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            txt_set.append(text_output.last_hidden_state[:, 0, :])

        txt_features = torch.cat(txt_set, 0).float()  # [image_num, feature_dim]
        rewards = self.mlp(txt_features)  # [image_num, 1]
        rewards = (rewards - self.mean) / self.std
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1

        return (
            indices.detach().cpu().numpy().tolist(),
            rewards.detach().cpu().numpy().tolist(),
        )


def rm_load(
    name: str = "ImageReward-v1.0",
    download_root: str = None,
    med_config: str = None,
):
    """Load a ImageReward model

    Parameters
    ----------
    name : str
        A model name listed by `ImageReward.available_models()`, or the path to a model checkpoint containing the state_dict

    download_root: str
        path to download the model files; by default, it uses "~/.cache/ImageReward"

    Returns
    -------
    model : torch.nn.Module
        The ImageReward model
    """
    download_root = download_root or os.path.join(_MODELS_ROOT, "ImageReward")
    os.makedirs(download_root, exist_ok=True)

    if name in RM.utils._MODELS:
        model_path = RM.ImageReward_download(
            RM.utils._MODELS[name],
            download_root,
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found;")

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    # med_config
    if med_config is None:
        med_config = RM.ImageReward_download(
            "https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json",
            download_root,
        )

    model = IRSMC(med_config=med_config)
    msg = model.load_state_dict(state_dict, strict=False)
    print("checkpoint loaded")
    model.eval()

    return model
