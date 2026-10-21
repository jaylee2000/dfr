# Training and configuration

Run commands from the repository root. See [installation](installation.md) for environment setup.

## Paper configs

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

## 1. Prepare the FD reference set

DFR-FD needs a prepared reference set for its training encoders. RAM, DFR-DS
and DFR-MMD do not need one unless FD diagnostics are enabled. The reference is
generated once per task from the base model (`fd_reference_seed: 0`) and shared
by all configs and seeds of that task. It is 25 × 1,152 base-model images,
about 8 A100 GPU-hours per task, and about 23 GB on disk.

```bash
source tools/install/cuda126_env.sh
uv run --no-sync -- accelerate launch --multi_gpu --num_processes 4 \
    --module dfr.fd_prepare geneval/dfr_fd
uv run --no-sync -- accelerate launch --multi_gpu --num_processes 4 \
    --module dfr.fd_prepare ocr/dfr_fd
```

The cache lands in `models/fd_stats/<hash>/`, keyed by the sampling settings
(model, resolution, steps, guidance, prompts, epoch sizes, reference seed).
A killed run resumes at the missing epochs. Preparation defaults to the active
FD spaces; `--spaces` selects an explicit list. Runs using FD refuse to start
without a matching reference and name the command to run.

## 2. Train

```bash
source tools/install/cuda126_env.sh
uv run --no-sync -- accelerate launch --multi_gpu --num_processes 4 \
    --module dfr.training_sd3 geneval/dfr_mmd
```

Any config key can be overridden with trailing `--key value` pairs (values are
parsed as YAML), for example a different β or seed:

```bash
uv run --no-sync -- accelerate launch --multi_gpu --num_processes 4 \
    --module dfr.training_sd3 ocr/dfr_ds --reverse_kl_weight 3 --seed 1 \
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
  the test prompts at each of `eval_guidance_scales`, together with ImageReward.
  These CFG values remain configurable through YAML or CLI overrides. A separate
  DreamSim diversity pass is opt-in and always uses CFG 4.5.
- **Stopping early.** `--stop_after_epoch N` stops after the checkpoint of
  epoch `N` (a multiple of `eval_every_n_epochs`).
- **W&B.** Off by default. Pass `--wandb_project <name>` (and optionally
  `--wandb_entity <team>`, or set `WANDB_ENTITY`) to log metrics and validation
  images.

## Optional diagnostics

By default, training loads only the encoders its reward needs: none for RAM or
DFR-MMD, the DINO patch branch for DFR-DS, and the configured FD training spaces
for DFR-FD. Scalar gradient norms and parameter-update statistics remain enabled.

Select extra measurements independently through YAML or CLI overrides:

| Setting | Effect |
| --- | --- |
| `fd_log_types` | additional FD spaces; default `[]` |
| `fd_log_every_n_epochs` | interval for those spaces; `0` disables them |
| `dreamsim_log_types` | extra DreamSim branches; default `[]` |
| `dreamsim_log_every_n_epochs` | training measurement interval for those branches; `0` disables them |
| `dreamsim_eval_num_prompts` | separate diversity-validation pass at CFG 4.5 for the selected branches; `0` disables it |

For example, to measure OpenCLIP diversity every five training epochs:

```bash
uv run --no-sync -- accelerate launch --module dfr.training_sd3 geneval/ram \
    --dreamsim_log_types '[open_clip_vitb32]' --dreamsim_log_every_n_epochs 5
```

FD diagnostics also need a prepared reference for their selected spaces. Every
measured FD space reports residual covariance distance (`fd_resid`), mean distance
(`fd_mean`), and pooled statistics (`fd`, `fd_cov`, `fd_ratio`) together. The
pooled null-FD used to normalize DFR-FD rewards is unchanged.
Keep the active FD space set fixed when resuming an existing checkpoint; use a
new output directory to change that set.

## Slurm

`scripts/slurm/train.sbatch` is a single-node template. Edit its `#SBATCH`
lines (account, partition, GPUs, time) once, then submit from the repository
root; arguments after the config pass through to training:

```bash
sbatch scripts/slurm/train.sbatch geneval/dfr_fd --seed 1
```

## Configuration reference

The task defaults document every key; each DFR module's docstring gives the
full derivation. The keys a user is most likely to change:

| Key | Meaning |
| --- | --- |
| `reverse_kl_weight` | β, the reverse-KL anchor of RAM |
| `reward_multiplier`, `scale_rewards` | task-reward scaling and normalization (`group`, `epoch`, `none`) |
| `dreamsim_scale` | DFR-DS weight, using the DINO ViT-B/16 patch branch without gating |
| `energy_scale`, `energy_lambda`, `energy_beta` | DFR-MMD weight, interaction coefficient λ and distance exponent b ∈ (0, 2] |
| `energy_reference_samples_per_prompt` | M, fresh base-model samples per prompt and epoch |
| `fd_scale`, `fd_train_types`, `fd_train_weights` | DFR-FD weight, feature spaces and per-space weights |
| `fd_log_types`, `dreamsim_log_types` | spaces and variants that are only logged, never optimized |

All three regularizers are added to the advantage after the task-reward
normalization and are centered within each prompt group. DFR-MMD uses
unit-normalized terminal latents. DFR-FD combines the mean and residual
covariance terms, normalizing each space by the base model's null-FD.
FD encoders:
`dinov2_vitl14`, `inception_v3`, `convnextv2_base`, `mae_vitl16`,
`siglip2_so400m`, `clip_vitl14`.

## Verify a prepared FD reference

```bash
uv run --no-sync -- python -m dfr.verify_fd_reference geneval/dfr_fd
```
