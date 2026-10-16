#!/usr/bin/env python3
import importlib.metadata
import platform

import torch


EXPECTED_DISTRIBUTIONS = {
    "mmcv": "2.1.0",
    "nvidia-nccl-cu12": "2.21.5",
    "paddlepaddle-gpu": "3.3.1",
    "torch": "2.6.0+cu126",
    "torchvision": "0.21.0+cu126",
    "vendi-score": "0.0.3",
}
SUPPORTED_CAPABILITIES = {(7, 5), (8, 0), (8, 9), (9, 0)}


def require_equal(label, actual, expected):
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, found {actual!r}")
    print(label, actual)


def main():
    require_equal("python", platform.python_version(), "3.10.16")
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        require_equal(distribution, importlib.metadata.version(distribution), expected)

    require_equal("torch.version.cuda", torch.version.cuda, "12.6")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA GPU")

    capability = torch.cuda.get_device_capability()
    if capability not in SUPPORTED_CAPABILITIES:
        raise RuntimeError(f"GPU compute capability {capability} is not in {sorted(SUPPORTED_CAPABILITIES)}")
    print("gpu", torch.cuda.get_device_name(), "compute capability", ".".join(map(str, capability)))
    print("torch cuda tensor", torch.ones(1, device="cuda").item())

    import torchvision  # noqa: F401
    import paddle
    import paddleocr  # noqa: F401
    import mmcv  # noqa: F401
    import mmcv._ext  # noqa: F401
    from mmcv.ops.roi_align import roi_align

    if not paddle.device.is_compiled_with_cuda() or paddle.device.cuda.device_count() == 0:
        raise RuntimeError("PaddlePaddle is not CUDA-ready or cannot see a GPU")
    paddle.set_device("gpu:0")
    print("paddle gpu tensor", float(paddle.ones([1], dtype="float32").numpy()[0]))

    feature = torch.arange(16, device="cuda", dtype=torch.float32).reshape(1, 1, 4, 4)
    rois = torch.tensor([[0, 0, 0, 3, 3]], device="cuda", dtype=torch.float32)
    aligned = roi_align(feature, rois, (2, 2), 1.0, 0, "avg", True)
    print("mmcv roi_align", aligned.detach().cpu().tolist())

    import accelerate  # noqa: F401
    import clip  # noqa: F401
    import clip_benchmark  # noqa: F401
    import diffusers  # noqa: F401
    import dreamsim  # noqa: F401
    import mmdet  # noqa: F401
    import mmengine  # noqa: F401
    import open_clip  # noqa: F401
    import peft  # noqa: F401
    import transformers  # noqa: F401
    import wandb  # noqa: F401
    from vendi_score import vendi  # noqa: F401
    import reward_models  # noqa: F401

    print("CUDA 12.6 environment validation passed")


if __name__ == "__main__":
    main()
