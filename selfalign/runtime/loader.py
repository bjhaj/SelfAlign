from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseModelHandle:
    """Opaque handle to a loaded base model (Phase 2 placeholder).

    Real model inference lands in Phase 6+; this is a deterministic stub to
    capture baselines and wire I/O.
    """
    model_id: str
    mode: str = "placeholder"
    device: str = "cpu"
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42


def load_base(model_id: str, device: str = "cuda") -> BaseModelHandle:
    """Return a deterministic placeholder handle for Phase 2.

    Args:
        model_id: Identifier for the base model (recorded only).
        device: Ignored for Phase 2; recorded on the handle for logging.
    """
    # Ignore actual HF loading in Phase 2
    dev = device or "cpu"
    return BaseModelHandle(model_id=model_id, device=dev)


def apply_adapter(handle: BaseModelHandle, adapter_dir: str) -> None:
    """Attach adapter to the base model (stub). No-op for Phase 2."""
    # Intentionally a no-op in placeholder mode.
    return None


def infer(handle: BaseModelHandle, prompt: str, max_new_tokens: int = 256) -> str:
    """Return a deterministic placeholder generation.

    Real model inference lands in Phase 6+; this is a deterministic stub to
    capture baselines and wire I/O.
    """
    # Controlled echo ensuring stability and bounded length
    prefix = "[BASE-PLACEHOLDER] "
    head = (prompt or "")[:200]
    return f"{prefix}{head} :: len={len(prompt or '')}"
