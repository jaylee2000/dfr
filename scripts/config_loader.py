"""Load training configs with explicit, shallow YAML inheritance."""

from pathlib import Path

import yaml


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _inside_configs(path):
    path = path.resolve()
    try:
        path.relative_to(CONFIGS_DIR.resolve())
    except ValueError as error:
        raise ValueError(f"Config path escapes {CONFIGS_DIR}: {path}") from error
    return path


def find_config(name):
    """Resolve a CLI config name: a path under configs/, or a bare task default."""
    # Append the extension as a string: Path.with_suffix would treat a dot in
    # the name (e.g. "..._scale9.09") as an existing suffix and truncate it.
    name = str(name)
    requested = Path(name if name.endswith(".yaml") else name + ".yaml")

    if requested.is_absolute():
        candidates = [requested]
    elif len(requested.parts) == 1:
        candidates = [CONFIGS_DIR / requested, CONFIGS_DIR / "defaults" / requested]
    else:
        candidates = [CONFIGS_DIR / requested]

    for candidate in candidates:
        candidate = _inside_configs(candidate)
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No config named {name!r}; searched {searched}")


def load_config_file(path, _stack=()):
    """Resolve ``extends`` relative to *path* and return the merged mapping."""
    path = _inside_configs(Path(path))
    if path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, path))
        raise ValueError(f"Config inheritance cycle: {chain}")

    with path.open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    data = dict(data)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str) or not parent:
        raise ValueError(f"extends must be a non-empty path string: {path}")

    parent_path = _inside_configs(path.parent / parent)
    if not parent_path.is_file():
        raise FileNotFoundError(f"Parent config does not exist: {parent_path}")
    merged = load_config_file(parent_path, (*_stack, path))
    merged.update(data)
    return merged


def load_config(name):
    return load_config_file(find_config(name))
