"""Reward modules used for RAM training and for the image-quality metrics
reported in the paper.

All scorers share the :class:`Reward` interface: they consume a float tensor of
images in ``[0, 1]`` with shape ``(B, C, H, W)`` and a list of prompts, and
return a per-sample score tensor on the same device as the images. Heavy
third-party dependencies are imported lazily inside ``__init__`` so that
importing this module does not pay for unused scorers.
"""

import json
from abc import ABC, abstractmethod

import torch
from PIL import Image
from torch import nn


class Reward(nn.Module, ABC):
    @abstractmethod
    def forward(self, images, prompts):
        """Score a batch of images.

        Args:
            images: float tensor in ``[0, 1]`` with shape ``(B, C, H, W)``.
            prompts: list of ``B`` text prompts.

        Returns:
            Float tensor of shape ``(B,)`` on ``images.device``.
        """


class AestheticScore(Reward):
    def __init__(self):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        state_dict = torch.load("models/sac+logos+ava1-l14-linearMSE.pth", map_location="cpu", weights_only=False)
        # Remove 'layers.' prefix from keys to match nn.Sequential structure
        state_dict = {k.replace("layers.", ""): v for k, v in state_dict.items()}
        self.mlp.load_state_dict(state_dict)

    def forward(self, images, prompts):
        device = images.device
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


class ImageReward(Reward):
    def __init__(self):
        super().__init__()
        from .image_reward import rm_load

        self.rm = rm_load()

    def forward(self, images, prompts):
        images = [x for x in images]
        return self.rm.score_from_prompt_batched(prompts, images)


class OCR(Reward):
    def __init__(self):
        super().__init__()
        from .ocr import OCRScorer

        self.scorer = OCRScorer()

    def forward(self, images, prompts):
        device = images.device
        np_images = (images * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        scores = self.scorer(list(np_images), prompts)
        return torch.tensor(scores, dtype=torch.float32, device=device)


class ClipScore(Reward):
    def __init__(self):
        super().__init__()
        from .clip_scorer import ClipScorer

        self.scorer = ClipScorer(device="cpu")

    def forward(self, images, prompts):
        return self.scorer(images, prompts)


class PickScore(Reward):
    def __init__(self):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path)

    def forward(self, images, prompts):
        device = images.device

        # Convert to uint8 PIL images for processor
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        images = images.permute(0, 2, 3, 1)  # (B, H, W, C)
        images = images.numpy(force=True)
        pil_images = [Image.fromarray(img) for img in images]

        # Preprocess images
        image_inputs = self.processor(
            images=pil_images, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

        # Preprocess text
        text_inputs = self.processor(text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # Get embeddings
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        # Calculate scores
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()

        # normalize to 0-1 range
        scores = scores / 26

        return scores


class Geneval(Reward):
    """
    Geneval reward using local Mask2Former + OpenCLIP evaluation (no reward server).

    Returns continuous scores for training (only_strict=True) or a full
    per-tag breakdown dict for eval logging (only_strict=False).
    """

    def __init__(self, metadata_path="prompts/geneval/metadata.jsonl"):
        super().__init__()
        from .gen_eval import GenevalEvaluator

        self.prompt_to_metadata: dict[str, dict] = {}
        with open(metadata_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                prompt = entry.get("prompt")
                if prompt:
                    self.prompt_to_metadata[prompt] = entry

        self.evaluator = GenevalEvaluator()

    def forward(self, images, prompts, only_strict=True):
        """
        Args:
            images: float tensor [0, 1] (B, C, H, W)
            prompts: list of strings (B,)
            only_strict: if True, return tensor of continuous scores;
                         if False, return dict[str, Tensor] with full breakdown.
        """
        metadata = []
        for prompt in prompts:
            meta = self.prompt_to_metadata.get(prompt)
            if meta is None:
                raise ValueError(
                    f"Prompt not found in geneval metadata: '{prompt}'. "
                    f"Make sure you're using prompts from geneval_train.txt or geneval_test.txt."
                )
            metadata.append(meta)

        self.evaluator.to(images.device)
        np_images = (images * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        pil_images = [Image.fromarray(img) for img in np_images]

        result = self.evaluator(pil_images, metadata, only_strict=only_strict)

        if only_strict:
            return torch.tensor(result, dtype=torch.float32, device=images.device).clamp(0.0, 1.0)
        return {k: torch.tensor(v, dtype=torch.float32, device=images.device) for k, v in result.items()}
