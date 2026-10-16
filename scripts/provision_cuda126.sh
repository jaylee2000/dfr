#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv >=0.11.28 is required." >&2
    exit 1
fi
if [[ ! -f uv.lock ]]; then
    echo "uv.lock is missing." >&2
    exit 1
fi

export RAM_SHARED_ROOT="${RAM_SHARED_ROOT:-${repo_root}/.ram-shared}"
mkdir -p "${RAM_SHARED_ROOT}/environments" "${RAM_SHARED_ROOT}/locks" "${RAM_SHARED_ROOT}/tmp" "${RAM_SHARED_ROOT}/uv-cache"

scripts/ensure_cuda126_toolchain.sh
# shellcheck source=scripts/cuda126_env.sh
source "${repo_root}/scripts/cuda126_env.sh"
unset VIRTUAL_ENV

environment_id="$(basename "${UV_PROJECT_ENVIRONMENT}")"
marker="${UV_PROJECT_ENVIRONMENT}/.ram-environment-complete"
exec 9>"${RAM_SHARED_ROOT}/locks/${environment_id}.lock"
flock 9

if [[ -f "${marker}" ]]; then
    echo "Reusing validated environment: ${UV_PROJECT_ENVIRONMENT}"
    exit 0
fi

artifact_dir="$(dirname "${RAM_MMCV_ARTIFACT}")"
artifact_name="$(basename "${RAM_MMCV_ARTIFACT}")"
(
    cd "${artifact_dir}"
    sha256sum --check "${artifact_name}.sha256"
)

uv python install 3.10.16
uv sync --locked --python 3.10.16 --no-install-package mmcv --no-build-package mmcv

bundle_dir="$(mktemp -d "${RAM_SHARED_ROOT}/tmp/mmcv-bundle.XXXXXX")"
cleanup() {
    rm -rf "${bundle_dir}"
}
trap cleanup EXIT

if tar -tzf "${RAM_MMCV_ARTIFACT}" | grep -E '(^/|(^|/)\.\.(/|$))' >/dev/null; then
    echo "Unsafe path found in MMCV artifact." >&2
    exit 1
fi
tar -xzf "${RAM_MMCV_ARTIFACT}" -C "${bundle_dir}" --no-same-owner

if [[ -f conda-cuda126-linux-64.lock ]]; then
    toolchain_lock="${repo_root}/conda-cuda126-linux-64.lock"
else
    toolchain_lock="${RAM_SHARED_ROOT}/conda-cuda126-linux-64.lock"
fi

wheel_path="$("${UV_PROJECT_ENVIRONMENT}/bin/python" scripts/mmcv_artifact.py verify \
    --bundle "${bundle_dir}" \
    --toolchain-lock "${toolchain_lock}")"
uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" --no-deps --reinstall "${wheel_path}"

"${UV_PROJECT_ENVIRONMENT}/bin/python" scripts/validate_cuda126.py

marker_tmp="${marker}.tmp.$$"
{
    echo "uv_lock_sha256=$(sha256sum uv.lock | awk '{print $1}')"
    echo "mmcv_artifact_sha256=$(awk 'NR == 1 {print $1}' "${RAM_MMCV_ARTIFACT}.sha256")"
    echo "toolchain_lock_sha256=$(sha256sum "${toolchain_lock}" | awk '{print $1}')"
} >"${marker_tmp}"
mv "${marker_tmp}" "${marker}"

echo "Provisioned validated environment: ${UV_PROJECT_ENVIRONMENT}"
echo "Use: uv run --no-sync -- <command>"
