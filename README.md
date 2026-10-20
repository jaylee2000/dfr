# Distribution-based regularization for reward fine-tuning diffusion and flow models

Code for **DFR**: reward fine-tuning of Stable Diffusion 3.5 Medium with
[Reinforce Adjoint Matching (RAM)](https://arxiv.org/abs/2605.10759), regularized
toward the base model's *distribution* instead of (or on top of) a per-sample KL
anchor. Three regularizers are implemented:

| Variant | Regularizer | Config keys |
| --- | --- | --- |
| DFR-DS | DreamSim within-prompt diversity reward | `dreamsim_*` |
| DFR-MMD | Energy-score (MMD) distance to base-model samples | `energy_*` |
| DFR-FD | Fréchet distance to the base model in frozen feature spaces | `fd_*` |

The repository contains the training code, the eight configs behind the paper's
GenEval and OCR tables, and an offline evaluation pipeline (`evaluation/`).

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Model weights](#model-weights)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Repository layout

```
configs/
  defaults/{geneval,ocr}_sd3.yaml   complete task defaults (RAM, beta = 1)
  geneval/{ram,dfr_ds,dfr_mmd,dfr_fd}.yaml
  ocr/{ram,dfr_ds,dfr_mmd,dfr_fd}.yaml
prompts/                            training and validation prompts (GenEval, OCR)
reward_models/                      GenEval, OCR, ImageReward, PickScore
scripts/
  training_sd3.py                   training entry point (RAM loss, rewards, DFR terms)
  dreamsim_diversity.py             DFR-DS
  energy_scoring.py                 DFR-MMD
  fd_reward.py, fd_reference.py     DFR-FD
  fd_prepare.py                     one-off FD reference set
  prefetch_*.py                     model downloads for offline compute nodes
  provision_cuda126.sh, ...         environment setup (see Installation)
  slurm/train.sbatch                generic Slurm template
evaluation/                         offline generation and metrics (own environment)
```

`scripts/training_sd3.py` is self-contained and the natural place to start
reading.

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
   scripts/build_mmcv_artifact.sh
   ```

2. **Provision the environment** from `uv.lock`, also on a GPU node. This
   installs the locked packages plus the MMCV wheel and runs
   `scripts/validate_cuda126.py` (PyTorch, Paddle and MMCV CUDA checks).

   ```bash
   scripts/provision_cuda126.sh
   ```

3. **Use it.** In every shell or job:

   ```bash
   source scripts/cuda126_env.sh
   uv run --no-sync -- python scripts/training_sd3.py --help
   ```

   Never run `uv sync` inside jobs; provision once and use the environment
   read-only.

The environment directory is named after the hashes of `uv.lock`, the toolchain
lock (`conda-cuda126-linux-64.lock`) and the MMCV artifact, so a lockfile change
provisions a *new* environment and running jobs are unaffected. Re-run
`scripts/provision_cuda126.sh` after pulling a lockfile change; MMCV is not
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
source scripts/cuda126_env.sh
# SD3.5 Medium, Mask2Former, OpenCLIP ViT-L/14, ImageReward, DreamSim, FD encoders
uv run --no-sync -- python scripts/prefetch_geneval_models.py
# PaddleOCR (OCR task): initialize once from a single process
uv run --no-sync -- python -c 'from reward_models.ocr import OCRScorer; OCRScorer()'
```

Weights land in `models/` (symlink it to large storage if needed). OpenCLIP and
PaddleOCR use their libraries' default caches unless `RAM_OPEN_CLIP_CACHE_DIR`
or `RAM_PADDLEOCR_CACHE_DIR` is set (see `.env.example`). Do not let several
ranks perform a first download concurrently: they share temporary file names
and can corrupt each other's files.

The optional `PickScore` training reward is not used by the paper configs;
`scripts/prefetch_pickscore_models.py` downloads its models together with
SD3.5 Medium and ImageReward.

## Training

### Paper configs

Each config extends its task default and sets only what differs. β is the
reverse-KL coefficient (`reverse_kl_weight`); the defaults are the paper's
main-table values.

| Config | Method | β | Regularizer |
| --- | --- | ---: | --- |
| `geneval/ram` | RAM | 1 | – |
| `geneval/dfr_ds` | DFR-DS | 2 | DreamSim (DINO), λ_DS = 3 |
| `geneval/dfr_mmd` | DFR-MMD | 2 | energy score on latents, λ_MMD = 0.55, b = 1.75, M = 8 |
| `geneval/dfr_fd` | DFR-FD | 2 | FD in ConvNeXt-V2, SigLIP2, CLIP-L, φ_FD = 0.07 |
| `ocr/ram` | RAM | 2 | – |
| `ocr/dfr_ds` | DFR-DS | 5 | DreamSim (DINO), λ_DS = 4 |
| `ocr/dfr_mmd` | DFR-MMD | 4 | energy score on latents, λ_MMD = 1.5, b = 1.75, M = 8 |
| `ocr/dfr_fd` | DFR-FD | 4 | FD in ConvNeXt-V2, SigLIP2, CLIP-L, φ_FD = 0.09 |

### 1. Prepare the FD reference set

Every config logs FD to the base model in six feature spaces, so **every**
config needs the FD reference set first, not only DFR-FD. The reference is
generated once per task from the base model (`fd_reference_seed: 0`) and shared
by all configs and seeds of that task. It is 25 × 1,152 base-model images,
about 8 A100 GPU-hours per task, and about 23 GB on disk.

```bash
source scripts/cuda126_env.sh
uv run --no-sync -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/fd_prepare.py geneval/dfr_fd
uv run --no-sync -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/fd_prepare.py ocr/dfr_fd
```

The cache lands in `models/fd_stats/<hash>/`, keyed by the sampling settings
(model, resolution, steps, guidance, prompts, epoch sizes, reference seed).
A killed run resumes at the missing epochs. Training refuses to start without a
matching reference and names the command to run.

### 2. Train

```bash
source scripts/cuda126_env.sh
uv run --no-sync -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/training_sd3.py geneval/dfr_mmd
```

Any config key can be overridden with trailing `--key value` pairs (values are
parsed as YAML), for example a different β or seed:

```bash
uv run --no-sync -- accelerate launch --multi_gpu --num_processes 4 \
    scripts/training_sd3.py ocr/dfr_ds --reverse_kl_weight 3 --seed 1 \
    --output_dir outputs/ocr/dfr_ds_beta3_seed1
```

- **Outputs.** The resolved config and checkpoints go to `outputs/<config>`
  (or `--output_dir`). Re-running the same command resumes from `latest`.
- **GPU count.** Epoch sizes (`num_prompts_per_epoch`, `num_samples_per_prompt`)
  are totals across processes, so results do not depend on the number of GPUs.
  The number of processes must divide `num_prompts_per_epoch` (48).
- **Memory.** `loss_batch_size`, `sampling_batch_size` and `reward_batch_size`
  are per process and only trade memory for speed. The configs' values were
  chosen for 96 GB GPUs; lower them on smaller ones.
- **Validation.** Every `eval_every_n_epochs`, the task reward is measured on
  the test prompts at each of `eval_guidance_scales`, together with ImageReward
  and DreamSim diversity.
- **Stopping early.** `--stop_after_epoch N` stops after the checkpoint of
  epoch `N` (a multiple of `eval_every_n_epochs`).
- **W&B.** Off by default. Pass `--wandb_project <name>` (and optionally
  `--wandb_entity <team>`, or set `WANDB_ENTITY`) to log metrics and validation
  images.

### Slurm

`scripts/slurm/train.sbatch` is a single-node template. Edit its `#SBATCH`
lines (account, partition, GPUs, time) once, then submit from the repository
root; arguments after the config pass through to training:

```bash
sbatch scripts/slurm/train.sbatch geneval/dfr_fd --seed 1
```

## Evaluation

`evaluation/` generates images from SD3.5 checkpoints and scores them offline
with ImageReward, PickScore, per-prompt FD to the base model in Inception-v3
and DINOv2 ViT-L/14 features, and DreamSim diversity with DINO ViT-B/16 (patch)
and OpenCLIP ViT-B/32. It has its own uv environment and needs neither the
CUDA toolchain nor MMCV. Task scores (GenEval, OCR) are measured during
training validation.

### Setup

```bash
cd evaluation
cp local.env.example local.env   # set RAM_ROOT, EVAL_ROOT, OUTPUTS_ROOT
source scripts/eval_env.sh       # loads local.env; selects evaluation/.venv
uv sync --project "${EVAL_ROOT}" --locked --python 3.10.16
```

`RAM_ROOT` points to this repository and `EVAL_ROOT` to its `evaluation/`
directory. `OUTPUTS_ROOT` receives `manifests/`, `images/` and `metrics/`.
In later shells, `source scripts/eval_env.sh` selects the environment again.

Download the models once, on a node with internet access:

```bash
scripts/prepare_models.sh prepare
scripts/prepare_models.sh verify
```

Generation and metrics then run fully offline from the resulting lock file
(`evaluation/models/models.lock.json`). If you prepared assets with an older
version, run preparation again to add DINOv2 and OpenCLIP before verification.

### Your checkpoints

`configs/checkpoints.yaml` ships only the SD3.5 Medium base model (`base_sd35`).
Add trained checkpoints in `configs/checkpoints.local.yaml` (git-ignored), using
the `ram_full` loader on a checkpoint directory written by training:

```yaml
version: 1
checkpoints:
  ocr_dfr_ds:
    display_name: OCR DFR-DS
    base_model: sd35_medium
    source:
      path: /absolute/path/to/outputs/ocr/dfr_ds/latest
    loader: ram_full
    guidance_scale: 4.5
    default_dtype: fp16
```

### Generate and score

```bash
bash scripts/create_manifests.sh --prompts geneval drawbench ocr
bash scripts/run_generation.sh --checkpoints base_sd35 \
    --prompts geneval drawbench ocr
bash scripts/run_generation.sh --checkpoints ocr_dfr_ds --prompts ocr drawbench
bash scripts/run_metric.sh --image-path-root "${OUTPUTS_ROOT}/images" \
    --checkpoints ocr_dfr_ds --prompts ocr drawbench --base-checkpoint base_sd35 \
    --metrics pickscore imagereward fid dreamsim
```

Evaluate GenEval checkpoints on `geneval drawbench` and OCR checkpoints on
`ocr drawbench`. The base checkpoint needs images for every evaluated set.
PickScore remains a metric; it is not a prompt set.

Manifests fix 256 source entries per dataset using prompt seed `20260529`
and 64 image seeds (`0` through `63`) shared by every prompt and checkpoint.
The source files contain 2,212 GenEval entries, 1,000 DrawBench entries, and
1,018 OCR entries. Repeated source entries are preserved when sampling.
Generation defaults to 512×512, 20 steps, FP16 transformer/text-encoder
execution, and FP32 VAE decoding. Checkpoint catalogs specify guidance
(4.5 normally; use 1.0 for multi-reward DiffusionNFT). `--dtype` and
`--vae-dtype` explicitly override precision.

Generation is resumable with unchanged settings. Use a fresh output directory
when switching from older generation settings, or regenerate with
`--overwrite`. Metrics reject missing images and mismatched prompt/seed
groups. For development, reduce manifest counts with `--geneval-count`,
`--drawbench-count`, `--ocr-count`, and `--images-per-prompt`; pass the same
image count to generation. `--limit-prompts` selects complete prompt groups;
`--limit-images` must end at a complete group. Small development runs do not
implement the full paper protocol.

Outputs live under `OUTPUTS_ROOT/metrics/<metric>/<checkpoint>/<dataset>/`.
ImageReward and PickScore write per-image and per-prompt scores. The historical
`fid` command writes per-prompt FD and summaries in `inception_v3/` and
`dinov2_vitl14/`, plus a combined encoder summary. FD uses the empirical
covariance with denominator N−1, compares each prompt's 64 images against
that prompt's base images, and averages the prompt distances.

DreamSim writes the `dino_vitb16_patch` and `open_clip_vitb32` variants in
per-prompt records and summaries. Each reports mean pairwise `1 − cosine`
distance over distinct images. The DINO patch branch concatenates the CLS
embedding with the spatially averaged patch embeddings. Divide these scores
by two to compare with training-time `dreamsim_diversity` logs.

## Configuration reference

The task defaults document every key; each DFR module's docstring gives the
full derivation. The keys a user is most likely to change:

| Key | Meaning |
| --- | --- |
| `reverse_kl_weight` | β, the reverse-KL anchor of RAM |
| `reward_multiplier`, `scale_rewards` | task-reward scaling and normalization (`group`, `epoch`, `none`) |
| `dreamsim_scale`, `dreamsim_train_type` | DFR-DS weight and DreamSim variant (`dino_vitb16_patch`, `ensemble`, ..., or `latent`) |
| `dreamsim_gate` | `none`, `solved` or `std`: switch the diversity term off for prompt groups whose task reward is constant |
| `energy_scale`, `energy_lambda`, `energy_beta` | DFR-MMD weight, interaction coefficient λ and distance exponent b ∈ (0, 2] |
| `energy_feature_extractor` | `latent`, `dreamsim:<variant>` or `fd:<encoder>` |
| `energy_reference_samples_per_prompt` | M, fresh base-model samples per prompt and epoch |
| `fd_scale`, `fd_train_types`, `fd_train_weights` | DFR-FD weight, feature spaces and per-space weights |
| `fd_normalize` | per-space normalization: `null_ratio` (default; divide by the base model's null-FD), `sg_ratio` or `none` |
| `fd_log_types`, `dreamsim_log_types` | spaces and variants that are only logged, never optimized |

All three regularizers are added to the advantage after the task-reward
normalization and are centered within each prompt group. FD encoders:
`dinov2_vitl14`, `inception_v3`, `convnextv2_base`, `mae_vitl16`,
`siglip2_so400m`, `clip_vitl14`. When `inception_v3` is loaded, the Inception
Vendi Score is logged as well.

## Troubleshooting

- **`CUDA 12.6 toolchain not found`**: provisioning has not run for this
  `RAM_SHARED_ROOT`, or `.env` is not loaded in this shell.
- **`MMCV artifact checksum not found`**: run `scripts/build_mmcv_artifact.sh`
  once before provisioning.
- **Hugging Face 401/403 errors**: set `HF_TOKEN` in `.env`. Importing the
  ImageReward scorer points `HF_HOME` at `models/.huggingface`, which can hide a
  token saved by `huggingface-cli login`.
- **Corrupted downloads or `FileNotFoundError` in a model cache**: a first
  download ran in several processes at once. Delete the partial cache and
  prefetch from a single process (see [Model weights](#model-weights)).
- **`The FD reference pass needs at least 2 x d samples`**: the epoch shape was
  shrunk; keep `fd_init_epochs × num_prompts_per_epoch × num_samples_per_prompt`
  at least 4,096 (twice Inception's 2,048 dimensions).
- **ARM64 hosts (e.g. GH200)**: the lock targets x86_64. Paddle has no ARM64
  CUDA wheel; `reward_models/ocr.py` falls back to running OCR on the CPU.

## Citation

```bibtex
@misc{passaro2026dfr,
  title  = {Distribution-based regularization for reward fine-tuning diffusion and flow models},
  author = {Passaro, Romeo and Lee, Jeongjae and Ye, Jong Chul and Bronstein, Michael M. and Bengio, Yoshua and Tong, Alexander and Rehman, Danyal},
  year   = {2026}
}
```

This code builds on RAM; please also cite:

```bibtex
@misc{bergmeister2026ram,
  title         = {Reinforce Adjoint Matching: Scaling RL Post-Training of Diffusion and Flow-Matching Models},
  author        = {Bergmeister, Andreas and Jegelka, Stefanie and N{\"u}sken, Nikolas and Domingo-Enrich, Carles and Pidstrigach, Jakiw},
  year          = {2026},
  eprint        = {2605.10759},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## Acknowledgements

We built on top of [RAM](https://arxiv.org/abs/2605.10759) and its original
implementation by Andreas Bergmeister. The GenEval and OCR rewards are adapted
from [Flow-GRPO](https://github.com/yifan123/flow_grpo) and
[DiffusionNFT](https://github.com/zhengkw18/DiffusionNFT). All prompt sets come
from Flow-GRPO (MIT, © 2025 Jie Liu): the OCR and evaluation prompt files are
byte-identical copies of its `dataset/` files, and the GenEval training files
are derived from its GenEval metadata. The DreamSim, ImageReward, PickScore and
OpenCLIP wrappers use the official packages.
The six FD feature spaces follow Yang et al., *Representation Fréchet Loss for
Visual Generation*.

## License

MIT; see [LICENSE](LICENSE).
