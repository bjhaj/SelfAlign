from __future__ import annotations


def train_sft(base_model_id: str, data_path: str, persona_yaml: str, out_dir: str, seed: int, use_qlora: bool = True) -> dict:
    """Train LoRA/QLoRA adapter from JSONL SFT data (stub).

    Args:
        base_model_id: Base model identifier.
        data_path: JSONL training data path.
        persona_yaml: Persona YAML path.
        out_dir: Output directory for adapter.
        seed: Random seed.
        use_qlora: Whether to use QLoRA.
    Returns:
        Summary metrics dict.
    """
    # TODO: Implement training loop and checkpointing
    return {"ok": True, "adapter_dir": out_dir, "steps": 0}
