#!/usr/bin/env bash

# Source this file before running RAM. The host driver remains untouched (any
# driver that supports CUDA 12.6 works); these variables select the shared CUDA
# 12.6 build toolchain and the uv environment built from the current lockfile.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this script instead of executing it: source tools/install/cuda126_env.sh" >&2
    exit 2
fi

_ram_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RAM_SHARED_ROOT="${RAM_SHARED_ROOT:-${_ram_repo_root}/.ram-shared}"
export RAM_CUDA126_PREFIX="${RAM_CUDA126_PREFIX:-${RAM_SHARED_ROOT}/toolchains/cuda126-gcc114}"
# shellcheck source=tools/install/mmcv_artifact_config.sh
source "${_ram_repo_root}/tools/install/mmcv_artifact_config.sh"
export RAM_MMCV_ARTIFACT="${RAM_MMCV_ARTIFACT:-${RAM_SHARED_ROOT}/artifacts/mmcv/${RAM_MMCV_ARTIFACT_BASENAME}}"

if [[ ! -x "${RAM_CUDA126_PREFIX}/bin/nvcc" ]]; then
    echo "CUDA 12.6 toolchain not found at ${RAM_CUDA126_PREFIX}." >&2
    echo "Run tools/install/provision_cuda126.sh once from a login or build node." >&2
    unset _ram_repo_root
    return 1
fi

export CUDA_HOME="${RAM_CUDA126_PREFIX}"
export CUDA_PATH="${RAM_CUDA126_PREFIX}"
export PATH="${RAM_CUDA126_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${RAM_CUDA126_PREFIX}/lib:${RAM_CUDA126_PREFIX}/lib64:${RAM_CUDA126_PREFIX}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CPATH="${RAM_CUDA126_PREFIX}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${RAM_CUDA126_PREFIX}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export CC="${RAM_CUDA126_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${RAM_CUDA126_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="${CXX}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${RAM_SHARED_ROOT}/uv-cache}"

if [[ -n "${RAM_UV_ENVIRONMENT:-}" ]]; then
    export UV_PROJECT_ENVIRONMENT="${RAM_UV_ENVIRONMENT}"
elif [[ -f "${_ram_repo_root}/uv.lock" ]]; then
    if [[ ! -f "${RAM_MMCV_ARTIFACT}.sha256" ]]; then
        echo "MMCV artifact checksum not found at ${RAM_MMCV_ARTIFACT}.sha256." >&2
        echo "Build it once with tools/install/build_mmcv_artifact.sh." >&2
        unset _ram_repo_root
        return 1
    fi
    _ram_lock_hash="$(sha256sum "${_ram_repo_root}/uv.lock" | awk '{print substr($1, 1, 16)}')"
    _ram_artifact_hash="$(awk 'NR == 1 {print substr($1, 1, 16)}' "${RAM_MMCV_ARTIFACT}.sha256")"
    if [[ -f "${_ram_repo_root}/conda-cuda126-linux-64.lock" ]]; then
        _ram_toolchain_lock="${_ram_repo_root}/conda-cuda126-linux-64.lock"
    else
        _ram_toolchain_lock="${RAM_SHARED_ROOT}/conda-cuda126-linux-64.lock"
    fi
    if [[ ! -f "${_ram_toolchain_lock}" ]]; then
        echo "Conda toolchain lock not found." >&2
        unset _ram_repo_root _ram_lock_hash _ram_artifact_hash _ram_toolchain_lock
        return 1
    fi
    _ram_toolchain_hash="$(sha256sum "${_ram_toolchain_lock}" | awk '{print substr($1, 1, 16)}')"
    export UV_PROJECT_ENVIRONMENT="${RAM_SHARED_ROOT}/environments/cu126-${_ram_lock_hash}-toolchain-${_ram_toolchain_hash}-mmcv-${_ram_artifact_hash}"
    unset _ram_lock_hash _ram_artifact_hash _ram_toolchain_hash _ram_toolchain_lock
fi

unset _ram_repo_root
