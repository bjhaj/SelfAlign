from __future__ import annotations

from typing import Any, List, Dict
from pathlib import Path


def read_yaml(path: str) -> dict:
    """Read a YAML file into a Python dict.

    Ensures the top-level is a mapping; returns {} for empty files.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required to load YAML files. Please install 'pyyaml'.") from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping/object at the top level.")
    return data


def write_yaml(obj: dict, path: str) -> None:
    """Write a Python dict to a YAML file using safe_dump.

    Creates parent directories as needed.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required to write YAML files. Please install 'pyyaml'.") from e

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=True, allow_unicode=True)


def read_jsonl(path: str) -> list[dict]:
    """Read JSONL file into list of dicts (stub)."""
    # TODO: Implement JSONL reader
    raise NotImplementedError


def write_jsonl(rows: list[dict], path: str) -> None:
    """Write rows to JSONL (stub)."""
    # TODO: Implement JSONL writer
    raise NotImplementedError


# --- Helpers ---

def load_yaml_dict(path: str) -> dict:
    """Load a YAML file and ensure the top-level is a mapping (dict).

    Args:
        path: Path to a YAML file.
    Returns:
        The parsed YAML content as a dict.
    Raises:
        ValueError: If the YAML cannot be parsed or the top-level is not a mapping.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required to load YAML files. Please install 'pyyaml'.") from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping/object at the top level.")
    return data
