# Troubleshooting

- **`CUDA 12.6 toolchain not found`**: provisioning has not run for this
  `RAM_SHARED_ROOT`, or `.env` is not loaded in this shell.
- **`MMCV artifact checksum not found`**: run `tools/install/build_mmcv_artifact.sh`
  once before provisioning.
- **Hugging Face 401/403 errors**: set `HF_TOKEN` in `.env`. Importing the
  ImageReward scorer points `HF_HOME` at `models/.huggingface`, which can hide a
  token saved by `huggingface-cli login`.
- **Corrupted downloads or `FileNotFoundError` in a model cache**: a first
  download ran in several processes at once. Delete the partial cache and
  prefetch from a single process (see [Model weights](installation.md#model-weights)).
- **`The FD reference pass needs at least 2 x d samples`**: the epoch shape was
  shrunk; keep `fd_init_epochs × num_prompts_per_epoch × num_samples_per_prompt`
  at least 4,096 (twice Inception's 2,048 dimensions).
- **ARM64 hosts (e.g. GH200)**: the lock targets x86_64. Paddle has no ARM64
  CUDA wheel; `dfr/reward_models/ocr.py` falls back to running OCR on the CPU.
