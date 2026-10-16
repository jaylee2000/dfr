#!/usr/bin/env bash

# Source this file before running the evaluation pipeline. It loads local.env,
# puts the RAM checkout and this directory on PYTHONPATH, and selects this
# directory's own uv environment (default: .venv). It needs no CUDA toolkit:
# generation and scoring use the CUDA runtime bundled with the PyTorch wheels.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script instead of executing it: source scripts/eval_env.sh" >&2
  exit 2
fi

_eval_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_eval_saved_root_dir="${ROOT_DIR-}"
_eval_root_dir_was_set="${ROOT_DIR+x}"
ROOT_DIR="${_eval_root}"
# shellcheck source=scripts/load_local_env.sh
source "${_eval_root}/scripts/load_local_env.sh"
if [[ -n "${_eval_root_dir_was_set}" ]]; then
  ROOT_DIR="${_eval_saved_root_dir}"
else
  unset ROOT_DIR
fi

case ":${PYTHONPATH:-}:" in
  *":${RAM_ROOT}:"*) ;;
  *) export PYTHONPATH="${RAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
case ":${PYTHONPATH:-}:" in
  *":${EVAL_ROOT}:"*) ;;
  *) export PYTHONPATH="${EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${EVAL_ROOT}/.venv}"
unset VIRTUAL_ENV
unset _eval_root _eval_saved_root_dir _eval_root_dir_was_set
