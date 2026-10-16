#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/eval_env.sh
source "${ROOT_DIR}/scripts/eval_env.sh"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

exec "${UV_PROJECT_ENVIRONMENT}/bin/python" -m ram_eval.generation "$@"
