#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/eval_env.sh
source "${ROOT_DIR}/scripts/eval_env.sh"

exec "${UV_PROJECT_ENVIRONMENT}/bin/python" -m ram_eval.prompts "$@"
