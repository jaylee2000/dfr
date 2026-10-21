#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv >=0.11.28 is required." >&2
    exit 1
fi

export RAM_SHARED_ROOT="${RAM_SHARED_ROOT:-${repo_root}/.ram-shared}"
tools/install/ensure_cuda126_toolchain.sh
# shellcheck source=tools/install/mmcv_artifact_config.sh
source "${repo_root}/tools/install/mmcv_artifact_config.sh"

lock_hash="$(sha256sum uv.lock | awk '{print substr($1, 1, 16)}')"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    # Compiling against Torch's ~7,000 generated headers on network scratch can
    # produce transient ENOENT failures. Keep the high-I/O build entirely on
    # node-local storage and publish only the completed artifact.
    local_build_root="${SLURM_TMPDIR}/ram-mmcv-artifact"
else
    local_build_root="${RAM_SHARED_ROOT}/build-local/mmcv-artifact"
    echo "Warning: SLURM_TMPDIR is unset; MMCV compilation is not on node-local storage." >&2
fi
export RAM_UV_ENVIRONMENT="${RAM_MMCV_BUILDER_ENVIRONMENT:-${local_build_root}/environment}"
shared_toolchain="${RAM_CUDA126_PREFIX:-${RAM_SHARED_ROOT}/toolchains/cuda126-gcc114}"
local_toolchain=""
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    # The compiler also opens thousands of GCC, glibc and CUDA headers from the
    # toolchain, which fail transiently on network scratch in the same way.
    # Compile against a node-local copy; the artifact references neither path.
    local_toolchain="${local_build_root}/toolchain"
    mkdir -p "${local_toolchain}"
    cp -a "${shared_toolchain}/." "${local_toolchain}/"
    export RAM_CUDA126_PREFIX="${local_toolchain}"
fi
export UV_LINK_MODE=copy
# shellcheck source=tools/install/cuda126_env.sh
source "${repo_root}/tools/install/cuda126_env.sh"
unset VIRTUAL_ENV

artifact_dir="${RAM_SHARED_ROOT}/artifacts/mmcv"
artifact_path="${artifact_dir}/${RAM_MMCV_ARTIFACT_BASENAME}"
build_root="${local_build_root}/mmcv-${RAM_MMCV_VERSION}-${lock_hash}"
source_archive="${build_root}/mmcv-${RAM_MMCV_VERSION}.tar.gz"
source_dir="${build_root}/source/mmcv-${RAM_MMCV_VERSION}"
wheel_dir="${build_root}/wheels"
mkdir -p "${artifact_dir}" "${build_root}" "${wheel_dir}" "${RAM_SHARED_ROOT}/locks" "${RAM_SHARED_ROOT}/uv-cache"

exec 9>"${RAM_SHARED_ROOT}/locks/mmcv-artifact-${RAM_MMCV_VERSION}.lock"
flock 9

if [[ -f "${artifact_path}" && -f "${artifact_path}.sha256" ]] && (
    cd "${artifact_dir}" && sha256sum --check "${RAM_MMCV_ARTIFACT_BASENAME}.sha256"
); then
    echo "Canonical MMCV artifact already exists: ${artifact_path}"
    env -u RAM_UV_ENVIRONMENT -u UV_PROJECT_ENVIRONMENT \
        RAM_SHARED_ROOT="${RAM_SHARED_ROOT}" \
        RAM_CUDA126_PREFIX="${shared_toolchain}" \
        tools/install/provision_cuda126.sh
    exit 0
fi

uv python install 3.10.16
uv sync --locked --python 3.10.16 --no-install-package mmcv --no-build-package mmcv

if [[ ! -f "${source_archive}" ]]; then
    curl --fail --location --output "${source_archive}.tmp" "${RAM_MMCV_SDIST_URL}"
    echo "${RAM_MMCV_SDIST_SHA256}  ${source_archive}.tmp" | sha256sum --check
    mv "${source_archive}.tmp" "${source_archive}"
fi
echo "${RAM_MMCV_SDIST_SHA256}  ${source_archive}" | sha256sum --check

if [[ ! -f "${source_dir}/setup.py" ]]; then
    source_stage="${build_root}/source.tmp.$$"
    rm -rf "${source_stage}"
    mkdir -p "${source_stage}"
    tar -xzf "${source_archive}" -C "${source_stage}" --no-same-owner
    mkdir -p "$(dirname "${source_dir}")"
    rm -rf "${source_dir}"
    mv "${source_stage}/mmcv-${RAM_MMCV_VERSION}" "${source_dir}"
    rmdir "${source_stage}"
fi

export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${RAM_MMCV_ARCH_LIST}"
export MAX_JOBS="${RAM_MMCV_MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

rm -f "${wheel_dir}"/mmcv-*.whl
(
    cd "${source_dir}"
    "${UV_PROJECT_ENVIRONMENT}/bin/python" setup.py bdist_wheel --dist-dir "${wheel_dir}"
)

wheel_path="$(find "${wheel_dir}" -maxdepth 1 -type f -name 'mmcv-2.1.0-*.whl' -print -quit)"
if [[ -z "${wheel_path}" ]]; then
    echo "MMCV wheel was not produced." >&2
    exit 1
fi

uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" --no-deps --reinstall "${wheel_path}"
extension_path="$("${UV_PROJECT_ENVIRONMENT}/bin/python" -c 'import mmcv._ext; print(mmcv._ext.__file__)')"

readelf_output="${build_root}/mmcv-extension-readelf.txt"
readelf --dynamic "${extension_path}" >"${readelf_output}"
if grep -F -e "${UV_PROJECT_ENVIRONMENT}" -e "${build_root}" ${local_toolchain:+-e "${local_toolchain}"} "${readelf_output}" >/dev/null; then
    echo "MMCV extension contains a non-relocatable build path." >&2
    exit 1
fi

cuobjdump_output="${build_root}/mmcv-extension-cuobjdump.txt"
"${CUDA_HOME}/bin/cuobjdump" --list-elf "${extension_path}" >"${cuobjdump_output}"
for architecture in sm_75 sm_80 sm_89 sm_90; do
    if ! grep -F "${architecture}" "${cuobjdump_output}" >/dev/null; then
        echo "MMCV extension is missing ${architecture}." >&2
        exit 1
    fi
done

"${UV_PROJECT_ENVIRONMENT}/bin/python" tools/install/validate_cuda126.py

if [[ -f conda-cuda126-linux-64.lock ]]; then
    toolchain_lock="${repo_root}/conda-cuda126-linux-64.lock"
else
    toolchain_lock="${RAM_SHARED_ROOT}/conda-cuda126-linux-64.lock"
fi

bundle_stage="$(mktemp -d "${build_root}/bundle.XXXXXX")"
cleanup() {
    rm -rf "${bundle_stage}"
}
trap cleanup EXIT
mkdir -p "${bundle_stage}/wheels"
cp "${wheel_path}" "${bundle_stage}/wheels/"
"${UV_PROJECT_ENVIRONMENT}/bin/python" tools/install/mmcv_artifact.py create \
    --wheel "${bundle_stage}/wheels/$(basename "${wheel_path}")" \
    --output "${bundle_stage}/manifest.json" \
    --uv-lock "${repo_root}/uv.lock" \
    --toolchain-lock "${toolchain_lock}" \
    --source-url "${RAM_MMCV_SDIST_URL}" \
    --source-sha256 "${RAM_MMCV_SDIST_SHA256}"
(
    cd "${bundle_stage}"
    sha256sum manifest.json wheels/*.whl >SHA256SUMS
)

artifact_tmp="${artifact_path}.tmp.$$"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
    -czf "${artifact_tmp}" -C "${bundle_stage}" manifest.json SHA256SUMS wheels
artifact_hash="$(sha256sum "${artifact_tmp}" | awk '{print $1}')"
echo "${artifact_hash}  ${RAM_MMCV_ARTIFACT_BASENAME}" >"${artifact_path}.sha256.tmp.$$"
mv "${artifact_tmp}" "${artifact_path}"
mv "${artifact_path}.sha256.tmp.$$" "${artifact_path}.sha256"

echo "Published canonical MMCV artifact: ${artifact_path}"

# Provision a fresh artifact-keyed environment. This verifies wheel relocation
# and leaves the shared runtime environment ready for subsequent jobs.
env -u RAM_UV_ENVIRONMENT -u UV_PROJECT_ENVIRONMENT \
    RAM_SHARED_ROOT="${RAM_SHARED_ROOT}" \
    RAM_CUDA126_PREFIX="${shared_toolchain}" \
    tools/install/provision_cuda126.sh
