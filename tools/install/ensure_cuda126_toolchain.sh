#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export RAM_SHARED_ROOT="${RAM_SHARED_ROOT:-${repo_root}/.ram-shared}"
export RAM_CUDA126_PREFIX="${RAM_CUDA126_PREFIX:-${RAM_SHARED_ROOT}/toolchains/cuda126-gcc114}"

if [[ -x "${RAM_CUDA126_PREFIX}/bin/nvcc" ]]; then
    exit 0
fi

conda_exe="${RAM_CONDA_EXE:-${CONDA_EXE:-}}"
if [[ -z "${conda_exe}" ]] && command -v conda >/dev/null 2>&1; then
    conda_exe="$(command -v conda)"
fi
if [[ -z "${conda_exe}" && -x "${HOME}/miniconda3/bin/conda" ]]; then
    conda_exe="${HOME}/miniconda3/bin/conda"
fi
if [[ ! -x "${conda_exe}" ]]; then
    echo "Conda executable not found. Set RAM_CONDA_EXE=/path/to/conda." >&2
    exit 1
fi

mkdir -p "${RAM_SHARED_ROOT}/toolchains" "${RAM_SHARED_ROOT}/locks"
exec 9>"${RAM_SHARED_ROOT}/locks/toolchain.lock"
flock 9

if [[ -x "${RAM_CUDA126_PREFIX}/bin/nvcc" ]]; then
    exit 0
fi

if [[ -f conda-cuda126-linux-64.lock ]]; then
    "${conda_exe}" create --yes --prefix "${RAM_CUDA126_PREFIX}" --file conda-cuda126-linux-64.lock
elif [[ -f "${RAM_SHARED_ROOT}/conda-cuda126-linux-64.lock" ]]; then
    "${conda_exe}" create --yes --prefix "${RAM_CUDA126_PREFIX}" --file "${RAM_SHARED_ROOT}/conda-cuda126-linux-64.lock"
else
    "${conda_exe}" env create --prefix "${RAM_CUDA126_PREFIX}" --file environment.cuda126-toolchain.yml
    "${conda_exe}" list --prefix "${RAM_CUDA126_PREFIX}" --explicit >"${RAM_SHARED_ROOT}/conda-cuda126-linux-64.lock"
    echo "Created ${RAM_SHARED_ROOT}/conda-cuda126-linux-64.lock." >&2
    echo "Copy it to the repository as conda-cuda126-linux-64.lock after validation." >&2
fi
