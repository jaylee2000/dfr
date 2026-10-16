#!/usr/bin/env bash
# Single-node RAM training on Slurm. Edit the #SBATCH lines for your cluster,
# then submit from the repository root:
#
#   sbatch scripts/slurm/train.sbatch <config> [--key value ...]
#   sbatch scripts/slurm/train.sbatch geneval/dfr_fd --seed 1
#
# Outputs go to outputs/<config>; re-submitting the same config resumes from
# its latest checkpoint. Epoch sizes are totals across GPUs, so the number of
# GPUs must divide num_prompts_per_epoch (48 in the paper configs).
#
#SBATCH --job-name=ram-train
#SBATCH --account=<your-account>
#SBATCH --partition=<gpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

config="${1:?Usage: sbatch scripts/slurm/train.sbatch <config> [--key value ...]}"
shift

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi
source scripts/cuda126_env.sh
export PYTHONUNBUFFERED=1  # stream training output to the Slurm log

num_gpus="${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus | wc -l)}"
launch_args=(--num_processes "${num_gpus}" --num_machines 1 --main_process_port "$((20000 + SLURM_JOB_ID % 20000))")
if (( num_gpus > 1 )); then
    launch_args+=(--multi_gpu)
fi

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${num_gpus} config=${config}"
srun --ntasks=1 uv run --no-sync -- \
    accelerate launch "${launch_args[@]}" scripts/training_sd3.py "${config}" "$@"
