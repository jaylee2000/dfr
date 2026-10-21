#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import zipfile


SCHEMA_VERSION = 1
EXPECTED_ARCHITECTURES = ["7.5", "8.0", "8.9", "9.0"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def torch_runtime():
    import torch

    return {
        "python": platform.python_version(),
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "torch": importlib.metadata.version("torch"),
        "torch_cuda": torch.version.cuda,
    }


def create_manifest(args):
    wheel = pathlib.Path(args.wheel).resolve()
    with zipfile.ZipFile(wheel) as archive:
        extension_names = sorted(name for name in archive.namelist() if name.startswith("mmcv/") and name.endswith(".so"))
    if not extension_names:
        raise SystemExit(f"{wheel} does not contain an MMCV extension")

    runtime = torch_runtime()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "package": "mmcv",
        "version": "2.1.0",
        **runtime,
        "platform": "linux-x86_64",
        "libc": list(platform.libc_ver()),
        "architectures": EXPECTED_ARCHITECTURES,
        "wheel": {"filename": wheel.name, "sha256": sha256(wheel), "extensions": extension_names},
        "source": {"url": args.source_url, "sha256": args.source_sha256},
        "uv_lock_sha256": sha256(args.uv_lock),
        "toolchain_lock_sha256": sha256(args.toolchain_lock),
        "nvcc": command_output([os.environ["CUDA_HOME"] + "/bin/nvcc", "--version"]),
        "compiler": command_output([os.environ["CC"], "--version"]).splitlines()[0],
    }
    pathlib.Path(args.output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def verify_bundle(args):
    bundle = pathlib.Path(args.bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit("Unsupported MMCV artifact manifest schema")
    if manifest.get("package") != "mmcv" or manifest.get("version") != "2.1.0":
        raise SystemExit("MMCV artifact package/version mismatch")
    if manifest.get("architectures") != EXPECTED_ARCHITECTURES:
        raise SystemExit("MMCV artifact architecture mismatch")
    if manifest.get("platform") != "linux-x86_64" or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("MMCV artifact platform mismatch")

    runtime = torch_runtime()
    for key in ("python", "python_abi", "torch", "torch_cuda"):
        if manifest.get(key) != runtime[key]:
            raise SystemExit(f"MMCV artifact {key} mismatch: {manifest.get(key)!r} != {runtime[key]!r}")

    toolchain_hash = sha256(args.toolchain_lock)
    if manifest.get("toolchain_lock_sha256") != toolchain_hash:
        raise SystemExit("MMCV artifact toolchain lock mismatch")

    wheel = bundle / "wheels" / manifest["wheel"]["filename"]
    if not wheel.is_file() or sha256(wheel) != manifest["wheel"]["sha256"]:
        raise SystemExit("MMCV wheel is missing or has the wrong checksum")
    print(wheel)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--wheel", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--uv-lock", required=True)
    create.add_argument("--toolchain-lock", required=True)
    create.add_argument("--source-url", required=True)
    create.add_argument("--source-sha256", required=True)
    create.set_defaults(func=create_manifest)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--toolchain-lock", required=True)
    verify.set_defaults(func=verify_bundle)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
