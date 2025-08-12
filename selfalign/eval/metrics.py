from __future__ import annotations


def style_score(text: str, persona_cfg: dict) -> float:
    """Compute style score (stub)."""
    # TODO: Implement embedding or rule-based metric
    return 0.0


def refusal_rate(texts: list[str]) -> float:
    """Compute refusal rate over texts (stub)."""
    # TODO: Implement refusal detection aggregate
    return 0.0


def toxicity_stats(texts: list[str]) -> dict:
    """Compute toxicity statistics (stub)."""
    # TODO: Implement simple proxy stats
    return {"mean": 0.0, "p95": 0.0}


def neutrality_score(text: str) -> float:
    """Compute neutrality drift score (stub)."""
    # TODO: Implement simple polarity heuristic
    return 0.0
