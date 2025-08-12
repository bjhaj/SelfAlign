"""SelfAlign Evaluation Harness — Phase 2 (Frozen Evaluation Set)

Purpose:
- Define and validate a frozen prompt set (JSONL) tagged by categories.
- Capture a baseline of generations (from a fixed/base model) for later
  comparison against trained adapters (style/refusal/safety/neutrality deltas).

Notes:
- This module intentionally avoids heavy runtime deps (no transformers here).
- Downstream metrics (style/refusal/toxicity/neutrality) will be added in later
  phases; here we focus on IO, schema, and baseline capture scaffolding.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, TypedDict, List, Dict, Any
import json
from pathlib import Path
from datetime import datetime, timezone
import statistics as _stats

from selfalign.eval.metrics import (
    refusal_detect,
    safety_violation_detect,
    sanitize_for_log,
)
from selfalign.utils.seed import set_global_seed
from selfalign.runtime.loader import load_base, infer as runtime_infer
from selfalign.utils.logging import new_run_dir, write_json
from selfalign.utils.io import file_sha256

# ---------------------------------------------------------------------------
# Tags and item schema
# ---------------------------------------------------------------------------
TAGS_ALLOWED = {"style", "qa", "refusal", "safety", "neutrality"}


class EvalItem(TypedDict):
    """Evaluation item schema.

    Fields:
      - id: unique identifier (string)
      - prompt: the user prompt (string)
      - tags: list of tags from TAGS_ALLOWED
    """

    id: str
    prompt: str
    tags: List[str]


@dataclass
class GenerationParams:
    """Generation parameter bundle used for baseline capture.

    Keep minimal and model-agnostic for portability.
    """

    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 256
    seed: int = 0


# Default generation parameters for Phase 2
DEFAULT_GEN_PARAMS = GenerationParams(temperature=0.7, top_p=0.9, max_new_tokens=256, seed=42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedupe_preserve_order_str(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

# Deterministic placeholder generation (no model dependency)
def _generate_output_placeholder(prompt: str, seed: int, idx: int) -> str:
    """Return a benign, deterministic placeholder response.

    Uses seed and item index to pick a stable template. Never produces
    actionable or refusal phrasing; suitable for baseline scaffolding.
    """
    templates = [
        "Here is a brief, neutral response to your prompt.",
        "Thanks for the prompt; here is a concise, general reply.",
        "Acknowledged. Providing a short, informative answer.",
        "This is a placeholder response for baseline capture.",
    ]
    # Simple deterministic index from seed and idx
    k = (seed * 31 + idx * 17) % len(templates)
    return templates[k]


def _percentile_nearest_rank(values: List[int], pct: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    n = len(vs)
    # Nearest-rank method
    k = max(1, int((pct / 100.0) * n + 0.999999))  # ceil without import
    k = min(k, n)
    return float(vs[k - 1])


def _print_quick_stats(items: List[EvalItem]) -> None:
    # Tag counts
    order = ["style", "qa", "refusal", "safety", "neutrality"]
    counts = {t: 0 for t in order}
    for it in items:
        for t in it.get("tags", []):
            if t in counts:
                counts[t] += 1
    # Prompt lengths
    lens = [len(it.get("prompt", "")) for it in items]
    mean_len = _stats.mean(lens) if lens else 0.0
    std_len = _stats.pstdev(lens) if len(lens) > 1 else 0.0
    p95_len = _percentile_nearest_rank(lens, 95.0) if lens else 0.0
    # Sample IDs
    sample_ids = [it.get("id", "?") for it in items[:3]]

    print("\nQuick evalset stats")
    print("-------------------")
    print("Counts by tag:")
    for t in order:
        print(f"  {t:<10} {counts[t]:>4}")
    print("Prompt length:")
    print(f"  mean={mean_len:.1f}  std={std_len:.1f}  p95={p95_len:.0f}")
    print(f"First 3 ids: {', '.join(sample_ids)}\n")


# ---------------------------------------------------------------------------
# Public API (Phase 2 skeleton)
# ---------------------------------------------------------------------------

def load_eval_set(path: str) -> List[Dict[str, Any]]:
    """Load and strictly validate a JSONL evaluation set.

    Each line must be a JSON object with keys:
      - id: non-empty string and unique across the file
      - prompt: non-empty string
      - tags: list[str] with 1..3 items (content validated later)

    Returns the list of dicts on success; raises ValueError with aggregated
    errors if any violations are found.
    """
    items: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen_ids: set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                errors.append(f"ERROR: line {ln}: invalid JSON: {e}")
                continue
            if not isinstance(obj, dict):
                errors.append(f"ERROR: line {ln}: must be a JSON object")
                continue

            # Field checks
            idv = obj.get("id")
            prm = obj.get("prompt")
            tags = obj.get("tags")

            if not isinstance(idv, str) or not idv.strip():
                errors.append(f"ERROR: line {ln}: /id must be a non-empty string")
            else:
                if idv in seen_ids:
                    errors.append(f"ERROR: line {ln}: duplicate id '{idv}'")
                else:
                    seen_ids.add(idv)
            if not isinstance(prm, str) or not prm.strip():
                errors.append(f"ERROR: line {ln}: /prompt must be a non-empty string")
            if not isinstance(tags, list):
                errors.append(f"ERROR: line {ln}: /tags must be a list of strings (length 1..3)")
            else:
                if not (1 <= len(tags) <= 3):
                    errors.append(f"ERROR: line {ln}: /tags length must be within 1..3")
                else:
                    for j, t in enumerate(tags):
                        if not isinstance(t, str) or not t.strip():
                            errors.append(f"ERROR: line {ln}: /tags[{j}] must be a non-empty string")

            items.append(obj)

    if errors:
        raise ValueError("\n".join(errors))
    return items


def validate_eval_items(items: List[Dict[str, Any]]) -> List[EvalItem]:
    """Validate and normalize eval items; return typed list or raise ValueError.

    Strict checks (aggregated):
    - ids unique (global)
    - all tags ∈ TAGS_ALLOWED
    - prompt length within 1..2000 chars
    - normalize tags: lowercase and dedupe preserving order
    """
    errors: List[str] = []
    out: List[EvalItem] = []

    seen_ids: set[str] = set()
    allowed_str = "{" + ",".join(f"'{t}'" for t in sorted(TAGS_ALLOWED)) + "}"

    for i, obj in enumerate(items):
        path = f"/items[{i}]"
        if not isinstance(obj, dict):
            errors.append(f"ERROR: {path}: must be an object")
            continue
        idv = obj.get("id")
        prm = obj.get("prompt")
        tags = obj.get("tags")

        # id unique
        if not isinstance(idv, str) or not idv.strip():
            errors.append(f"ERROR: {path}/id must be a non-empty string")
        else:
            if idv in seen_ids:
                errors.append(f"ERROR: duplicate id '{idv}'")
            else:
                seen_ids.add(idv)

        # prompt length
        if not isinstance(prm, str) or not prm.strip():
            errors.append(f"ERROR: {path}/prompt must be a non-empty string")
        else:
            if not (1 <= len(prm) <= 2000):
                errors.append(
                    f"ERROR: {path}/prompt length {len(prm)} out of bounds [1,2000]"
                )

        # tags allowed and normalize
        norm_tags: List[str] = []
        if not isinstance(tags, list) or not tags:
            errors.append(f"ERROR: {path}/tags must be a non-empty list of strings")
        else:
            for j, t in enumerate(tags):
                if not isinstance(t, str) or not t.strip():
                    errors.append(f"ERROR: {path}/tags[{j}] must be a non-empty string")
                    continue
                tok = t.strip().lower()
                if tok not in TAGS_ALLOWED:
                    errors.append(
                        f"ERROR: {path}/tags[{j}] '{t}' not in {allowed_str}"
                    )
                else:
                    norm_tags.append(tok)
            norm_tags = _dedupe_preserve_order_str(norm_tags)

        if not errors:
            out.append(EvalItem(id=idv, prompt=prm, tags=norm_tags))

    if errors:
        raise ValueError("\n".join(errors))
    return out


def run_baseline_capture(
    evalset_path: str,
    out_report: str,
    gen_params: GenerationParams,
    *,
    expect_non_refusal_on: Optional[set[str]] = None,
    keep_safety_gate: bool = True,
    non_refusal_min: Optional[float] = None,
    unsafe_leak_max: Optional[float] = None,
) -> Dict[str, Any]:
    # Ensure deterministic placeholder generation as well
    set_global_seed(gen_params.seed)

    # Load and validate eval set
    raw = load_eval_set(evalset_path)
    items = validate_eval_items(raw)

    # Load base placeholder handle (Phase 2)
    handle = load_base("llm:open-8b", device="auto")

    enable_non_refusal = bool(expect_non_refusal_on and ("refusal" in expect_non_refusal_on))
    refusal_flags: List[tuple[bool, bool]] = []  # (refused, unsafe)

    # Collect generations and write samples JSONL under a new run dir
    run_dir = new_run_dir("baseline")
    samples_dir = Path(run_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    samples_path = samples_dir / "base_golden.jsonl"

    items_out: List[Dict[str, Any]] = []
    with open(samples_path, "w", encoding="utf-8") as sf:
        for it in items:
            output = runtime_infer(handle, it["prompt"], max_new_tokens=gen_params.max_new_tokens)

            analyze_row: Dict[str, Any] = {"id": it["id"], "prompt": it["prompt"], "output": output}
            sample_row: Dict[str, Any] = {"id": it["id"], "prompt": it["prompt"], "output": sanitize_for_log(output)}

            # Optional analysis for refusal-tagged items
            if enable_non_refusal and ("refusal" in it["tags"]):
                refused = bool(refusal_detect(output))
                unsafe = bool(safety_violation_detect(output)) if keep_safety_gate else False
                refusal_flags.append((refused, unsafe))
                analyze_row["refused"] = refused
                analyze_row["unsafe_leak"] = unsafe
                sample_row["refused"] = refused
                sample_row["unsafe_leak"] = unsafe

            items_out.append(analyze_row)
            sf.write(json.dumps(sample_row, ensure_ascii=False) + "\n")

    # Aggregates for refusal slice
    analysis: Optional[Dict[str, Any]] = None
    if enable_non_refusal:
        total = len(refusal_flags)
        if total == 0:
            non_refusal_rate: Any = "n/a"
            unsafe_leak_rate: Any = "n/a"
        else:
            non_refusal_rate = sum(1 for r, _ in refusal_flags if not r) / total
            unsafe_leak_rate = sum(1 for _, u in refusal_flags if u) / total
        notes = "Safety gate enforced" if keep_safety_gate else "Safety gate disabled"
        analysis = {
            "expected_behavior": "non_refusal",
            "total_items": total,
            "non_refusal_rate": non_refusal_rate,
            "unsafe_leak_rate": unsafe_leak_rate,
            "thresholds": {
                "non_refusal_min": non_refusal_min,
                "unsafe_leak_max": unsafe_leak_max,
            },
            "notes": notes,
        }
        # Console summary
        def fmt_pct(x: Any) -> str:
            return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "n/a"
        nr_pct = fmt_pct(non_refusal_rate)
        ul_pct = fmt_pct(unsafe_leak_rate)
        nr_thresh_txt = (
            f" (PASS >={non_refusal_min * 100:.0f}%)" if isinstance(non_refusal_min, (int, float)) and isinstance(non_refusal_rate, (int, float)) and (non_refusal_rate >= non_refusal_min) else
            (f" (FAIL <{non_refusal_min * 100:.0f}%)" if isinstance(non_refusal_min, (int, float)) and isinstance(non_refusal_rate, (int, float)) else "")
        )
        ul_thresh_txt = (
            f" (PASS <={unsafe_leak_max * 100:.0f}%)" if isinstance(unsafe_leak_max, (int, float)) and isinstance(unsafe_leak_rate, (int, float)) and (unsafe_leak_rate <= unsafe_leak_max) else
            (f" (FAIL >{unsafe_leak_max * 100:.0f}%)" if isinstance(unsafe_leak_max, (int, float)) and isinstance(unsafe_leak_rate, (int, float)) else "")
        )
        print(f"[refusal-tag] non-refusal: {nr_pct}{nr_thresh_txt}")
        print(f"              unsafe leaks: {ul_pct}{ul_thresh_txt}")

    # Build report structure
    ts = datetime.now(timezone.utc).isoformat()
    report: Dict[str, Any] = {
        "ok": True,
        "system": "base:placeholder",
        "evalset": evalset_path,
        "generation_params": asdict(gen_params),
        "timestamp": ts,
        "items": items_out,
    }
    if analysis is not None:
        report["refusal_tag_analysis"] = analysis

    # Ensure parent directories for out_report
    out_path = Path(out_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write pretty JSON report
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Console quick stats
    _print_quick_stats(items)

    # Freeze guard sidecar: record SHA256 of evalset and report
    try:
        eval_hash = file_sha256(evalset_path)
        report_hash = file_sha256(str(out_path))
        sidecar = Path(out_path).with_suffix(".hash")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with open(sidecar, "w", encoding="utf-8") as hf:
            hf.write(f"evalset_sha256 {eval_hash}\n")
            hf.write(f"report_sha256  {report_hash}\n")
        print("Recorded evalset and report hashes. Changing the golden file invalidates deltas.")
    except Exception as e:
        print(f"WARN: failed to write hash sidecar: {e}")

    return report


# ---------------------------------------------------------------------------
# Back-compat stub retained for CLI in Milestone 1
# ---------------------------------------------------------------------------

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
    # TODO: Load eval set, compute metrics, write report (Phase 6)
    return {"ok": True, "adapter": adapter_path or "base"}
