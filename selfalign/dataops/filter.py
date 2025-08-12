from __future__ import annotations


def apply_filters(records: list[dict], persona_cfg: dict) -> list[dict]:
    """Apply hard + smart filters (stub)."""
    # TODO: Implement basic validations and optional smart filters
    return records


def compute_style_score(text: str, persona_cfg: dict) -> float:
    """Compute style adherence proxy (stub)."""
    # TODO: Implement embedding or rule-based heuristic
    return 0.0


def refusal_detect(text: str) -> bool:
    """Detect refusal patterns (stub)."""
    # TODO: Implement regex-based detector
    return False


def toxicity_proxy(text: str) -> float:
    """Lightweight toxicity proxy (stub)."""
    # TODO: Implement lexicon or tiny model proxy
    return 0.0
