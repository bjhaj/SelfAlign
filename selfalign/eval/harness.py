from __future__ import annotations

from typing import Optional


def run_eval(adapter_path: str | None, evalset_path: str, persona_yaml: Optional[str], out_report: str) -> dict:
    """Run evaluation and return metrics dict (stub).

    Args:
        adapter_path: Path to adapter dir or None for base.
        evalset_path: Path to JSONL eval set.
        persona_yaml: Optional persona YAML for style metrics.
        out_report: Path to write detailed report JSON.
    Returns:
        Summary metrics dict.
    """
    # TODO: Load eval set, compute metrics, write report
    return {"ok": True, "adapter": adapter_path or "base"}
