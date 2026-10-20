"""Download every model needed by RAM GenEval and FD-reference generation."""

import gc
import os
import urllib.request
from pathlib import Path

from dfr import fd_encoders
from tools.download.prefetch_dreamsim_models import prefetch_dreamsim_models


REPO_ROOT = Path(__file__).resolve().parents[2]
# Training and the reward models read model files from <repo>/models.
MODELS = REPO_ROOT / "models"
MASK2FORMER_NAME = "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth"
MASK2FORMER_URL = (
    "https://download.openmmlab.com/mmdetection/v3.0/mask2former/"
    "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/"
    + MASK2FORMER_NAME
)


def open_clip_cache_kwargs():
    """Match GenEval: use an override when set, otherwise OpenCLIP's default."""
    cache_dir = os.environ.get("RAM_OPEN_CLIP_CACHE_DIR")
    return {"cache_dir": cache_dir} if cache_dir else {}


def download_atomic(url, path):
    """Download to a sibling temporary file, publishing only complete data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        urllib.request.urlretrieve(url, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    MODELS.mkdir(parents=True, exist_ok=True)

    print("[1/6] Stable Diffusion 3.5 Medium")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="stabilityai/stable-diffusion-3.5-medium",
        cache_dir=str(MODELS),
    )

    print("[2/6] GenEval Mask2Former")
    mask_path = MODELS / MASK2FORMER_NAME
    if not mask_path.exists():
        download_atomic(MASK2FORMER_URL, mask_path)

    print("[3/6] GenEval OpenCLIP ViT-L/14")
    import open_clip

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        device="cpu",
        **open_clip_cache_kwargs(),
    )
    del clip_model
    gc.collect()

    print("[4/6] ImageReward v1.0")
    from dfr.reward_models.image_reward import rm_load

    image_reward = rm_load()
    del image_reward
    gc.collect()

    print("[5/6] DreamSim models")
    prefetch_dreamsim_models(MODELS / "dreamsim")

    print("[6/6] FD reference encoders")
    fd_encoders.prefetch(MODELS / "fd_encoders")

    open_clip_cache = os.environ.get("RAM_OPEN_CLIP_CACHE_DIR", "OpenCLIP's default cache")
    print(
        f"GenEval assets are ready under {MODELS}, DreamSim under {MODELS / 'dreamsim'}, "
        f"and FD encoders under {MODELS / 'fd_encoders'}; "
        f"GenEval OpenCLIP cache: {open_clip_cache}"
    )


if __name__ == "__main__":
    main()
