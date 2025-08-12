from __future__ import annotations

import io
import os
from pathlib import Path
import tempfile
import pytest

from selfalign.persona.schema import validate_persona_yaml, lint_persona_files, last_warnings
from selfalign.utils.io import write_yaml


EXAMPLE_SOCRATIC = {
    "id": "socratic-skeptical",
    "version": "0.1.0",
    "description": "Questions assumptions, asks for evidence, pushes for clarity.",
    "values": {
        "candor": 0.8,
        "prudence": 0.6,
        "helpfulness": 0.7,
        "skepticism": 0.9,
        "risk_aversion": 0.5,
        "empathy": 0.4,
        "neutrality": 0.6,
    },
    "style": {
        "tone": "calm, precise, inquisitive",
        "verbosity": "medium",
        "structure": ["claim", "evidence", "counterpoint", "Next Steps"],
    },
    "taboos": [
        "make definitive claims without at least one source or caveat"
    ],
    "redlines": {
        "safety": ["illicit hard instructions", "biological threat details", "violent wrongdoing"],
        "privacy": ["PII extraction", "doxxing"],
    },
    "refusal_policy": {
        "band": [0.05, 0.15],
        "template": "I can’t help with that. Here’s a safer alternative: ...",
    },
}


def test_valid_personas_pass_and_normalize(tmp_path: Path):
    p = tmp_path / "socratic-skeptical.yaml"
    write_yaml(EXAMPLE_SOCRATIC, str(p))
    out = validate_persona_yaml(str(p))
    assert isinstance(out, dict)
    assert out.get("values_vector") and len(out["values_vector"]) == 7
    # structure should be kebab-case and contain next-steps
    assert "next-steps" in out["style"]["structure"]


def test_missing_value_key_raises(tmp_path: Path):
    obj = dict(EXAMPLE_SOCRATIC)
    obj["values"] = dict(obj["values"])  # copy
    del obj["values"]["skepticism"]
    p = tmp_path / "socratic-skeptical.yaml"
    write_yaml(obj, str(p))
    with pytest.raises(ValueError) as ei:
        validate_persona_yaml(str(p))
    assert "/values/skepticism" in str(ei.value)
    assert "is missing" in str(ei.value)


def test_refusal_band_low_gt_high(tmp_path: Path):
    obj = dict(EXAMPLE_SOCRATIC)
    obj["refusal_policy"] = dict(obj["refusal_policy"])
    obj["refusal_policy"]["band"] = [0.3, 0.2]
    p = tmp_path / "socratic-skeptical.yaml"
    write_yaml(obj, str(p))
    with pytest.raises(ValueError) as ei:
        validate_persona_yaml(str(p))
    assert "low" in str(ei.value) or "≤" in str(ei.value)


def test_banned_tone_word(tmp_path: Path):
    obj = dict(EXAMPLE_SOCRATIC)
    obj["style"] = dict(obj["style"])  # shallow copy
    obj["style"]["tone"] = "calm, friendly"
    p = tmp_path / "socratic-skeptical.yaml"
    write_yaml(obj, str(p))
    with pytest.raises(ValueError) as ei:
        validate_persona_yaml(str(p))
    assert "Banned vague descriptor" in str(ei.value)


def test_structure_normalization_warn(tmp_path: Path):
    obj = dict(EXAMPLE_SOCRATIC)
    obj["style"] = dict(obj["style"])  # shallow copy
    obj["style"]["structure"] = ["Next Steps", "claim"]
    p = tmp_path / "socratic-skeptical.yaml"
    write_yaml(obj, str(p))
    _ = validate_persona_yaml(str(p))
    warns = list(last_warnings)
    assert any("not kebab-case" in w for w in warns)


def test_roundtrip_stable_normalization(tmp_path: Path):
    p = tmp_path / "socratic-skeptical.yaml"
    write_yaml(EXAMPLE_SOCRATIC, str(p))
    canon1 = validate_persona_yaml(str(p))
    p2 = tmp_path / f"{canon1['id']}.normalized.yaml"
    write_yaml(canon1, str(p2))
    canon2 = validate_persona_yaml(str(p2))
    assert canon1 == canon2
