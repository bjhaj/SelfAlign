from __future__ import annotations

from typing import List, Dict


def generate_synthetic_data(persona_cfg: dict, n: int, seed: int) -> List[dict]:
    """Generate n synthetic SFT records (stub).

    Args:
        persona_cfg: Validated persona configuration.
        n: Number of records to generate.
        seed: Random seed.
    Returns:
        List of records (instruction, input, output, metadata).
    """
    # TODO: Implement persona-aware generation
    return []


def synthesize_to_file(persona_yaml: str, n: int, out_path: str, seed: int) -> Dict[str, int]:
    """Generate → filter (v0 hard only) → write JSONL; return summary dict (stub).

    Args:
        persona_yaml: Path to persona YAML.
        n: Number to generate.
        out_path: Output JSONL path.
        seed: Random seed.
    Returns:
        Summary dict with counts.
    """
    # TODO: Load YAML, generate data, apply hard filters, write JSONL
    return {"generated": n, "written": 0}
