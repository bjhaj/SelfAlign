from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseModelHandle:
    """Opaque handle to a loaded base model (stub)."""
    model_id: str


def load_base(model_id: str, device: str = "cuda") -> BaseModelHandle:
    """Load the base model (stub).

    Args:
        model_id: Identifier for the base model.
        device: Device string (e.g., cuda).
    Returns:
        BaseModelHandle instance.
    """
    # TODO: Implement actual model loading
    return BaseModelHandle(model_id=model_id)


def apply_adapter(handle: BaseModelHandle, adapter_dir: str) -> None:
    """Attach adapter to the base model (stub)."""
    # TODO: Implement adapter attach
    return None


def infer(handle: BaseModelHandle, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate text using the current adapter (stub)."""
    # TODO: Implement inference
    return "[stub output]"
