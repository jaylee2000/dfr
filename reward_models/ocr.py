"""Edit-distance OCR reward following the Flow-GRPO setup.

The reward extracts the quoted target text from each prompt (e.g. ``'a sign that
says "hello"'``) and compares it against PaddleOCR's recognition of the rendered
image. The score is ``1 - normalized_edit_distance``, clipped to ``[0, 1]``.
"""

import os

import numpy as np
import paddle
import torch
from Levenshtein import distance
from paddleocr import PaddleOCR
from PIL import Image


class OCRScorer:
    def __init__(self, use_gpu: bool = True):
        # Paddle has no CUDA wheel for ARM64 hosts (e.g. GH200). When CUDA
        # Paddle is absent, keep SD3/FD on the GPU and run the OCR recognizer
        # on the CPU.
        use_gpu = bool(use_gpu and paddle.is_compiled_with_cuda())
        paddleocr_args = {
            "use_angle_cls": False,
            "lang": "en",
            "use_gpu": use_gpu,
            "show_log": False,
        }

        # PaddleOCR 2.x otherwise hard-codes ~/.paddleocr. Set
        # RAM_PADDLEOCR_CACHE_DIR to keep its models in a shared cache instead.
        cache_root = os.environ.get("RAM_PADDLEOCR_CACHE_DIR")
        if cache_root:
            paddleocr_args.update(
                det_model_dir=os.path.join(
                    cache_root, "whl", "det", "en", "en_PP-OCRv3_det_infer"
                ),
                rec_model_dir=os.path.join(
                    cache_root, "whl", "rec", "en", "en_PP-OCRv4_rec_infer"
                ),
                cls_model_dir=os.path.join(
                    cache_root, "whl", "cls", "ch_ppocr_mobile_v2.0_cls_infer"
                ),
            )
        self.ocr = PaddleOCR(**paddleocr_args)

    @torch.no_grad()
    def __call__(
        self, images: list[Image.Image] | list[np.ndarray], prompts: list[str]
    ) -> list[float]:
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        # OCR prompts wrap the target text in double quotes; extract it.
        targets = [prompt.split('"')[1] for prompt in prompts]

        rewards = []
        for img, target in zip(images, targets):
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                result = self.ocr.ocr(img, cls=False)
                recognized = "".join(line[1][0] for line in result[0] if line[1][1] > 0) if result[0] else ""
                recognized = recognized.replace(" ", "").lower()
                target_norm = target.replace(" ", "").lower()
                if target_norm in recognized:
                    dist = 0
                else:
                    dist = min(distance(recognized, target_norm), len(target_norm))
            except Exception:
                dist = len(target)  # maximum penalty

            rewards.append(1 - dist / len(target))

        return rewards
