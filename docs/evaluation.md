# Offline evaluation

The setup below starts from the repository root and then enters `evaluation/`.

`evaluation/` generates images from SD3.5 checkpoints and scores them offline
with ImageReward, PickScore, per-prompt FD to the base model in Inception-v3
and DINOv2 ViT-L/14 features, and DreamSim diversity with DINO ViT-B/16 (patch)
and OpenCLIP ViT-B/32. It has its own uv environment and needs neither the
CUDA toolchain nor MMCV. Task scores (GenEval, OCR) are measured during
training validation.

## Setup

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

## Your checkpoints

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

## Generate and score

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
