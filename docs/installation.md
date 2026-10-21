# Installation and model weights

Run commands from the repository root. See the [README](../README.md) for basic usage.

## Installation

### Requirements

- Linux x86_64 with NVIDIA GPUs and a driver that supports CUDA 12.6 or newer
  (a CUDA 13 driver works).
- [uv](https://docs.astral.sh/uv/) `>=0.11.28,<0.12`.
- `conda`, `mamba` or `micromamba`, used once to create a CUDA 12.6 compiler
  prefix. It does not manage the Python environment.
- Disk: about 30 GB for the toolchain, the environment and the uv cache; about
  35 GB for model weights; about 23 GB per task for the FD reference set; and
  about 75 GB for the evaluation pipeline's models.
- A Hugging Face token with access to the gated
  [`stabilityai/stable-diffusion-3.5-medium`](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium).

The Python environment pins PyTorch 2.6.0 (cu126), PaddlePaddle GPU 3.3.1 and
MMCV 2.1.0. MMCV's CUDA ops (needed by the GenEval reward's object detector)
are compiled once into a binary artifact; normal installs never compile
anything.

### Setup

Run everything from the repository root.

```bash
cp .env.example .env      # set RAM_SHARED_ROOT, RAM_CONDA_EXE and HF_TOKEN
set -a; source .env; set +a
```

Load `.env` this way in every new shell; `scripts/slurm/train.sbatch` loads it
by itself. `RAM_SHARED_ROOT` must be visible from every compute node; it
defaults to `.ram-shared/` inside the repository.

1. **Build the MMCV artifact** once, on a GPU node (it ends by provisioning and
   validating an environment, which needs a GPU). It creates the CUDA 12.6 +
   GCC 11.4 toolchain first if it is missing, compiles MMCV 2.1.0 for compute
   capabilities 7.5, 8.0, 8.9 and 9.0 (RTX 8000, A100, L40S, H100), and writes a
   checksummed tarball to `$RAM_SHARED_ROOT/artifacts/mmcv/`. On Slurm it
   compiles on node-local `$SLURM_TMPDIR` against a local copy of the toolchain,
   because network filesystems can fail header reads under parallel builds.

   ```bash
   tools/install/build_mmcv_artifact.sh
   ```

2. **Provision the environment** from `uv.lock`, also on a GPU node. This
   installs the locked packages plus the MMCV wheel and runs
   `tools/install/validate_cuda126.py` (PyTorch, Paddle and MMCV CUDA checks).

   ```bash
   tools/install/provision_cuda126.sh
   ```

3. **Use it.** In every shell or job:

   ```bash
   source tools/install/cuda126_env.sh
   uv run --no-sync -- python -m dfr.training_sd3 --help
   ```

   Never run `uv sync` inside jobs; provision once and use the environment
   read-only.

The environment directory is named after the hashes of `uv.lock`, the toolchain
lock (`conda-cuda126-linux-64.lock`) and the MMCV artifact, so a lockfile change
provisions a *new* environment and running jobs are unaffected. Re-run
`tools/install/provision_cuda126.sh` after pulling a lockfile change; MMCV is not
recompiled. Old environments under `$RAM_SHARED_ROOT/environments/` can be
deleted once no job uses them.

To regenerate `uv.lock` (maintainers only), comment out
`no-build-package = ["mmcv"]` in `pyproject.toml` for the `uv lock` call and
restore it afterwards. Static MMCV metadata in `pyproject.toml` keeps uv from
building MMCV while resolving.

## Model weights

Compute nodes are often offline. Download every model once from a node with
internet access; all paths are relative to the repository root.

```bash
source tools/install/cuda126_env.sh
# SD3.5 Medium, Mask2Former, OpenCLIP ViT-L/14, ImageReward, DreamSim, FD encoders
uv run --no-sync -- python -m tools.download.prefetch_geneval_models
# PaddleOCR (OCR task): initialize once from a single process
uv run --no-sync -- python -c 'from dfr.reward_models.ocr import OCRScorer; OCRScorer()'
```

Weights land in `models/` (symlink it to large storage if needed). OpenCLIP and
PaddleOCR use their libraries' default caches unless `RAM_OPEN_CLIP_CACHE_DIR`
or `RAM_PADDLEOCR_CACHE_DIR` is set (see `.env.example`). Do not let several
ranks perform a first download concurrently: they share temporary file names
and can corrupt each other's files.

The optional `PickScore` training reward is not used by the paper configs;
`python -m tools.download.prefetch_pickscore_models` downloads its models together with
SD3.5 Medium and ImageReward.

## Optional DreamSim downloads

To populate only the DreamSim cache:

```bash
uv run --no-sync -- python -m tools.download.prefetch_dreamsim_models
```

The training distribution is named `dfr`. This metadata change updates `uv.lock`,
so provisioning selects a new environment through the existing lockfile hash.
The existing CUDA toolchain and validated MMCV artifact can be reused.
