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

## Installation

Requires Linux x86_64, NVIDIA GPUs with a CUDA 12.6-compatible driver,
[uv](https://docs.astral.sh/uv/) `>=0.11.28,<0.12`, and access to the gated
SD3.5 Medium model. Conda, mamba, or micromamba creates the compiler toolchain;
uv manages the Python environment.

From the repository root, configure shared storage, Conda, and your Hugging Face token:

```bash
cp .env.example .env
set -a; source .env; set +a
```

Build the MMCV artifact once on a GPU node, then select the environment:

```bash
tools/install/build_mmcv_artifact.sh
source tools/install/cuda126_env.sh
uv run --no-sync -- python -m dfr.training_sd3 --help
```

The build command also provisions and validates the locked environment. For an
existing artifact, run `tools/install/provision_cuda126.sh` instead. Run commands
from the repository root and source the environment in each new shell.

Download model weights once from an internet-connected node:

```bash
uv run --no-sync -- python -m tools.download.prefetch_geneval_models
# OCR: initialize its model cache once before launching multiple processes.
uv run --no-sync -- python -c 'from dfr.reward_models.ocr import OCRScorer; OCRScorer()'
```

See [installation and model weights](docs/installation.md) for requirements,
cache locations, offline preparation, and lockfile maintenance.

## Training

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

DFR-FD requires a base-model reference cache prepared once per task:

```bash
uv run --no-sync -- accelerate launch --multi_gpu --num_processes 4 \
    --module dfr.fd_prepare geneval/dfr_fd
```

Use `ocr/dfr_fd` for the OCR reference. RAM, DFR-DS, and DFR-MMD need no FD
reference unless optional FD diagnostics are enabled.

Launch training with a preset and optional YAML-valued overrides:

```bash
uv run --no-sync -- accelerate launch --multi_gpu --num_processes 4 \
    --module dfr.training_sd3 geneval/dfr_mmd --seed 1
```

Checkpoints go to `outputs/<config>` unless `--output_dir` is set. Re-running
resumes from `latest`. Epoch sizes are global; GPU count must divide the number
of prompts per epoch. Reduce per-process batch sizes for smaller GPUs.

Task and quality-score validation use configurable `eval_guidance_scales`.
Optional DreamSim validation always uses CFG 4.5. Extra training diagnostics are
disabled by default. PickScore remains available for training and evaluation.

For Slurm, edit the resource settings and submit from the repository root:

```bash
sbatch scripts/slurm/train.sbatch geneval/dfr_fd --seed 1
```

See [training and configuration](docs/configuration.md) for overrides, reference
verification, diagnostics, validation, and resume requirements.

## Evaluation

The independent evaluation environment generates images and measures
ImageReward, PickScore, per-prompt FD in Inception-v3 and DINOv2-L/14, and
DreamSim diversity in DINO patch and OpenCLIP features.

Follow [offline evaluation](docs/evaluation.md) to prepare the environment and
models, register checkpoints, and run generation and scoring. It documents the
paper's 256-prompt × 64-seed protocol and precision settings. GenEval and OCR
task scores are measured during training validation.

## Repository layout

- `dfr/`: training, regularizers, generation, FD references, and reward models.
- `configs/` and `prompts/`: paper presets and training/validation prompt data.
- `tools/install/` and `tools/download/`: environment setup and model preparation.
- `scripts/slurm/`: training submission template.
- `evaluation/`: offline generation and metrics with its own environment.
- `docs/`: detailed setup, configuration, evaluation, and [troubleshooting](docs/troubleshooting.md).

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
