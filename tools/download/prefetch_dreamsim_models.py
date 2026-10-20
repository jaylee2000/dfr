#!/usr/bin/env python3
"""Download every supported DreamSim checkpoint into a shared cache."""

import argparse
import gc
from pathlib import Path

from dfr.dreamsim_diversity import DREAMSIM_TYPES


DREAMSIM_MODEL_NAMES = tuple(DREAMSIM_TYPES)


def load_dreamsim(model_name, cache_dir):
    """Load one configured DreamSim variant on CPU to populate its cache."""
    from dreamsim import dreamsim

    dreamsim_type, use_patch_model = DREAMSIM_TYPES[model_name]
    return dreamsim(
        pretrained=True,
        dreamsim_type=dreamsim_type,
        use_patch_model=use_patch_model,
        device="cpu",
        cache_dir=str(cache_dir),
    )


def prefetch_dreamsim_models(cache_dir):
    """Load each model once on CPU so DreamSim downloads and verifies its files."""
    cache_dir = Path(cache_dir)
    for model_name in DREAMSIM_MODEL_NAMES:
        print(f"Prefetching DreamSim {model_name} into {cache_dir}")
        model, _ = load_dreamsim(model_name, cache_dir)
        del model
        gc.collect()
    print(f"DreamSim cache ready: {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default="models/dreamsim",
        help="DreamSim cache directory; must match the config key dreamsim_cache_dir (default: models/dreamsim)",
    )
    args = parser.parse_args()
    prefetch_dreamsim_models(args.cache_dir)


if __name__ == "__main__":
    main()
