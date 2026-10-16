"""Quickly validate a prepared FD reference without loading feature tensors."""

import argparse
import os
from argparse import Namespace

import fd_reference
import fd_reward
from config_loader import load_config
from fd_prepare import resolve_config as resolve_fd_config


def resolve_config(name, cache_dir=None, seed=None, fd_reference_seed=None):
    config = Namespace(**load_config(name))
    if seed is not None:
        config.seed = seed
    if fd_reference_seed is not None:
        config.fd_reference_seed = fd_reference_seed
    config = resolve_fd_config(config, int(os.environ.get("WORLD_SIZE", "1")))
    if cache_dir is not None:
        config.fd_cache_dir = cache_dir
    return config


def verify(config):
    directory = fd_reference.check_metadata(config)
    spaces = fd_reward.fd_loaded_types(config)
    fd_reference.load_reference_features(config, spaces, map_location="meta")
    fd_reference.load_null_fd(config, spaces)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        nargs="+",
        help="bare config names or paths relative to configs/, with or without .yaml",
    )
    parser.add_argument("--fd-cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fd-reference-seed", type=int, default=None)
    args = parser.parse_args()

    directories = []
    for name in args.configs:
        directory = verify(
            resolve_config(
                name,
                args.fd_cache_dir,
                seed=args.seed,
                fd_reference_seed=args.fd_reference_seed,
            )
        )
        directories.append(directory.resolve())
        print(f"PASS {name}: {directory}")

    if len(set(directories)) != 1:
        raise RuntimeError(f"Configs do not share one FD reference: {directories}")
    print(f"Shared FD reference ready: {directories[0]}")


if __name__ == "__main__":
    main()
