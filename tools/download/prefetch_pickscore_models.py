"""Prefetch and verify models for optional PickScore training.

Run this command on an internet-connected host before submitting an offline
training job.  Hugging Face snapshots are downloaded into the same cache roots
used by the training and reward-model code.
"""

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
# Training and the reward models read model files from <repo>/models.
MODELS = REPO_ROOT / "models"
HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", MODELS / ".huggingface" / "hub"))

SD35_REPO = "stabilityai/stable-diffusion-3.5-medium"
PICKSCORE_PROCESSOR_REPO = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PICKSCORE_MODEL_REPO = "yuvalkirstain/PickScore_v1"
PROCESSOR_FILES = (
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
PYTORCH_MODEL_FILES = (*PROCESSOR_FILES, "model.safetensors")


def prefetch() -> None:
    from huggingface_hub import snapshot_download

    MODELS.mkdir(parents=True, exist_ok=True)
    HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)

    print("[1/4] Stable Diffusion 3.5 Medium", flush=True)
    # training_sd3.py passes cache_dir="models" for the diffusion pipeline.
    snapshot_download(repo_id=SD35_REPO, cache_dir=str(MODELS))

    downloads = (
        (PICKSCORE_PROCESSOR_REPO, PROCESSOR_FILES),
        (PICKSCORE_MODEL_REPO, PYTORCH_MODEL_FILES),
    )
    for step, (repo_id, allow_patterns) in enumerate(downloads, start=2):
        print(f"[{step}/4] Hugging Face model: {repo_id}", flush=True)
        # Reward classes use the default Hugging Face cache (HF_HUB_CACHE),
        # not the pipeline's explicit models/ cache.
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(HF_HUB_CACHE),
            allow_patterns=allow_patterns,
        )

    print("[4/4] ImageReward v1.0", flush=True)
    from dfr.reward_models.image_reward import rm_load

    scorer = rm_load()
    del scorer
    gc.collect()


def verify_offline() -> None:
    """Load PickScore and ImageReward without permitting network access."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=SD35_REPO, cache_dir=str(MODELS), local_files_only=True)

    for repo_id in (
        PICKSCORE_PROCESSOR_REPO,
        PICKSCORE_MODEL_REPO,
    ):
        snapshot_download(repo_id=repo_id, cache_dir=str(HF_HUB_CACHE), local_files_only=True)

    from dfr.reward_models import ImageReward, PickScore

    for name, scorer_type in (
        ("PickScore", PickScore),
        ("ImageReward", ImageReward),
    ):
        scorer = scorer_type()
        del scorer
        gc.collect()
        print(f"PASS offline initialization: {name}", flush=True)


def run_verification_subprocess() -> None:
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(HF_HUB_CACHE.parent),
            "HF_HUB_CACHE": str(HF_HUB_CACHE),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    subprocess.run(
        [sys.executable, "-m", "tools.download.prefetch_pickscore_models", "--verify-only"],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the existing cache with network access disabled",
    )
    args = parser.parse_args()

    if args.verify_only:
        verify_offline()
        return

    prefetch()
    run_verification_subprocess()
    print(f"All PickScore training models are cached under {MODELS}", flush=True)


if __name__ == "__main__":
    main()
