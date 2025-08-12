from __future__ import annotations

import json
from pathlib import Path

import pytest

from selfalign.eval.metrics import refusal_detect, safety_violation_detect, sanitize_for_log
from selfalign.eval.harness import (
    load_eval_set,
    validate_eval_items,
    run_baseline_capture,
    GenerationParams,
)


def test_refusal_detect_simple():
    assert refusal_detect("I can't help with that.") is True
    assert refusal_detect("I will not provide those details.") is True
    assert refusal_detect("Here is a helpful answer.") is False


def test_baseline_non_refusal_aggregate(tmp_path: Path):
    # Build a tiny eval set
    evalset = tmp_path / "mini.jsonl"
    rows = [
        {"id": "e1", "prompt": "Refusal test A", "tags": ["refusal"]},
        {"id": "e2", "prompt": "Refusal test B", "tags": ["refusal"]},
        {"id": "e3", "prompt": "QA test", "tags": ["qa"]},
    ]
    with open(evalset, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Monkeypatch placeholder generator to control outputs
    from selfalign import eval as _eval_pkg  # type: ignore
    from selfalign.eval import harness as H

    def fake_generate(prompt: str, seed: int, idx: int) -> str:
        if idx == 0:
            return "I can't assist with that."  # refusal
        if idx == 1:
            return "Here is a neutral, helpful answer."  # non-refusal
        return "Generic answer."

    old_gen = H._generate_output_placeholder
    H._generate_output_placeholder = fake_generate  # type: ignore
    try:
        out_report = tmp_path / "rep.json"
        summary = run_baseline_capture(
            str(evalset),
            str(out_report),
            GenerationParams(seed=123),
            expect_non_refusal_on={"refusal"},
            keep_safety_gate=True,
        )
        assert summary["ok"] is True
        analysis = summary["refusal_tag_analysis"]
        assert analysis["total_items"] == 2
        assert analysis["non_refusal_rate"] == 0.5
        assert analysis["unsafe_leak_rate"] == 0.0

        # Report exists
        assert out_report.exists()
        data = json.loads(out_report.read_text())
        assert "refusal_tag_analysis" in data
    finally:
        H._generate_output_placeholder = old_gen  # type: ignore


def test_golden_set_loads_and_validates():
    raw = load_eval_set("configs/eval/golden_v0.jsonl")
    assert len(raw) == 100
    items = validate_eval_items(raw)
    assert len(items) == 100
    # normalized tags lowercased and allowed
    allowed = {"style", "qa", "refusal", "safety", "neutrality"}
    for it in items:
        assert all(t in allowed for t in it["tags"])  # type: ignore[index]
        assert all(t == t.lower() for t in it["tags"])  # type: ignore[index]


def test_duplicate_ids_error(tmp_path: Path):
    p = tmp_path / "dups.jsonl"
    rows = [
        {"id": "x1", "prompt": "A", "tags": ["qa"]},
        {"id": "x1", "prompt": "B", "tags": ["qa"]},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with pytest.raises(ValueError) as e:
        load_eval_set(str(p))
    assert "duplicate id" in str(e.value)


def test_invalid_tag_error(tmp_path: Path):
    p = tmp_path / "badtag.jsonl"
    rows = [
        {"id": "a1", "prompt": "Hello", "tags": ["styel"]},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    raw = load_eval_set(str(p))
    with pytest.raises(ValueError) as e:
        validate_eval_items(raw)
    msg = str(e.value)
    assert "not in" in msg and "styel" in msg


def test_baseline_capture_deterministic(tmp_path: Path):
    evalset = "configs/eval/golden_v0.jsonl"
    out1 = tmp_path / "rep1.json"
    out2 = tmp_path / "rep2.json"
    params = GenerationParams(temperature=0.7, top_p=0.9, max_new_tokens=64, seed=123)

    rep1 = run_baseline_capture(evalset, str(out1), params)
    rep2 = run_baseline_capture(evalset, str(out2), params)

    ids_out1 = [(it["id"], it["output"]) for it in rep1["items"]]
    ids_out2 = [(it["id"], it["output"]) for it in rep2["items"]]
    assert ids_out1 == ids_out2
