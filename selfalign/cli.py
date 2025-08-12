from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Iterable
import glob
import click

from selfalign.utils.logging import new_run_dir, log_event, write_json
from selfalign.utils.seed import set_global_seed
from selfalign.persona.schema import lint_persona_files, validate_persona_yaml
from selfalign.dataops.synth import synthesize_to_file
from selfalign.train.sft import train_sft
from selfalign.eval.harness import run_eval, run_baseline_capture, GenerationParams
from selfalign.runtime.loader import load_base, apply_adapter, infer
from selfalign.utils.io import write_yaml


@click.group()
def cli() -> None:
    """SelfAlign CLI (Milestone 1)."""


@cli.command()
def doctor() -> None:
    """Run environment checks and write a doctor.json into a new run dir."""
    run_dir = new_run_dir("doctor")
    payload = {
        "ok": True,
        "python": os.sys.version.split()[0],
        "cwd": str(Path.cwd()),
    }
    write_json(str(Path(run_dir) / "doctor.json"), payload)
    log_event(run_dir, "doctor", payload)
    click.echo(f"Doctor OK. Report: {run_dir}/doctor.json")


@cli.group()
def persona() -> None:
    """Persona-related commands."""


@persona.command("lint")
@click.argument("files", nargs=-1, type=str)
@click.option("--emit-normalized", "emit_norm_dir", type=click.Path(), help="Write <id>.normalized.yaml for OK files to OUTDIR")
@click.option("--json", "report_json", type=click.Path(), help="Write full JSON report to REPORT_PATH")
def persona_lint(files: tuple[str, ...], emit_norm_dir: Optional[str], report_json: Optional[str]) -> None:
    """Lint persona YAML files.

    Accepts multiple paths and glob patterns.
    Prints a table and exits non-zero if any errors are found.
    Optionally writes normalized YAML and a JSON report.
    """
    # Expand globs
    expanded: list[str] = []
    for pattern in files:
        matches = glob.glob(pattern)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(pattern)

    if not expanded:
        raise SystemExit("No files provided.")

    run_dir = new_run_dir("persona_lint")

    results = lint_persona_files(expanded)

    # Print header
    click.echo(f"{'Path':<60}  {'Status':<6}  {'#Errors':>7}  {'#Warnings':>9}")
    click.echo("-" * 90)

    any_errors = False
    report_rows: list[dict] = []

    # Optional normalized output directory
    out_dir = Path(emit_norm_dir) if emit_norm_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for path, ok, errors, warnings in results:
        status = "OK" if ok else "FAIL"
        any_errors = any_errors or (not ok)
        click.echo(f"{path:<60}  {status:<6}  {len(errors):>7}  {len(warnings):>9}")

        row = {
            "path": path,
            "ok": ok,
            "errors": errors,
            "warnings": warnings,
        }
        report_rows.append(row)

        # Emit normalized YAML if requested and OK
        if ok and out_dir is not None:
            try:
                canonical = validate_persona_yaml(path)
                out_path = out_dir / f"{canonical['id']}.normalized.yaml"
                write_yaml(canonical, str(out_path))
            except Exception as e:
                # Should not happen since ok=True, but guard anyway
                click.echo(f"Warning: failed to write normalized YAML for {path}: {e}")

    # Write run log JSON report
    report = {"results": report_rows}
    run_report_path = Path(run_dir) / "persona_lint.json"
    write_json(str(run_report_path), report)
    log_event(run_dir, "persona_lint", report)

    # Write explicit report if requested
    if report_json:
        write_json(report_json, report)

    if any_errors:
        raise SystemExit(1)
    raise SystemExit(0)


@cli.group()
def eval():
    """Evaluation utilities."""
    pass


@eval.command("baseline")
@click.option("--evalset", required=True, help="Path to evalset JSONL, e.g., configs/eval/golden_v0.jsonl")
@click.option("--out", required=True, help="Path to write report JSON, e.g., reports/base_golden.json")
@click.option("--max-new-tokens", default=256, show_default=True, type=int)
@click.option("--temperature", default=0.7, show_default=True, type=float)
@click.option("--top-p", default=0.95, show_default=True, type=float)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--base", "base_model", default=None, help="Base model ID (e.g., HF repo id). Defaults to $SELFALIGN_BASE_MODEL or dolphin-34b")
@click.option(
    "--base-backend",
    type=click.Choice(["placeholder", "transformers"], case_sensitive=False),
    default=None,
    help="Backend to use. Defaults to 'transformers' if $SELFALIGN_ENABLE_HF=1 else 'placeholder'",
)
@click.option("--device", "base_device", default="auto", show_default=True, help="Device string, e.g., 'auto', 'cuda', 'cpu'")
def eval_baseline(evalset, out, max_new_tokens, temperature, top_p, seed, base_model, base_backend, base_device):
    """Run baseline capture over the eval set and write a report."""
    params = GenerationParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    report = run_baseline_capture(
        evalset,
        out,
        params,
        base_model=base_model,
        base_backend=(base_backend.lower() if base_backend else None),
        base_device=base_device,
    )
    click.echo(f"Wrote report to {out}. Items: {len(report.get('items', []))}")


@eval.command("head")
@click.option("--report", required=True, help="Path to a baseline report JSON file")
@click.option("--n", default=3, show_default=True, type=int, help="Number of rows to preview")
def eval_head(report, n):
    """Print the first N rows: (id, prompt[:80], output[:80]) for quick inspection."""
    try:
        with open(report, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception as e:
        raise SystemExit(f"Failed to read report: {e}")

    items = report.get("items", []) or []
    n = max(0, min(n, len(items)))
    rows = items[:n]

    def trunc(s: object, width: int) -> str:
        txt = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
        txt = txt.replace("\n", " ").replace("\r", " ")
        if len(txt) <= width:
            return txt
        return txt[: width - 1] + "…"

    id_w = max(8, min(20, max((len(r.get("id", "")) for r in rows), default=8)))
    p_w = 80
    o_w = 80

    header = f"{'id':<{id_w}}  {'prompt':<{p_w}}  {'output':<{o_w}}"
    click.echo(header)
    click.echo("-" * len(header))

    for r in rows:
        idv = r.get("id", "")
        prompt = r.get("prompt", "")
        output = r.get("output", "")
        click.echo(f"{trunc(idv, id_w):<{id_w}}  {trunc(prompt, p_w):<{p_w}}  {trunc(output, o_w):<{o_w}}")


@cli.command()
@click.option("--adapter", "adapter_dir", required=True, type=click.Path())
def swap(adapter_dir: str) -> None:
    """Hot-swap adapter (stub)."""
    run_dir = new_run_dir("swap")
    handle = load_base("llm:open-8b")
    apply_adapter(handle, adapter_dir)
    write_json(str(Path(run_dir) / "swap.json"), {"ok": True, "adapter": adapter_dir})
    click.echo(f"Swap complete. Adapter: {adapter_dir}")


@cli.command()
@click.option("--adapter", "adapter_dir", required=True, type=click.Path())
@click.option("-p", "prompt", required=True, type=str)
@click.option("--max-new-tokens", default=256, type=int)
def infer_cmd(adapter_dir: str, prompt: str, max_new_tokens: int) -> None:
    """Run inference with currently attached adapter (stub)."""
    run_dir = new_run_dir("infer")
    handle = load_base("llm:open-8b")
    apply_adapter(handle, adapter_dir)
    output = infer(handle, prompt, max_new_tokens=max_new_tokens)
    write_json(str(Path(run_dir) / "infer.json"), {"prompt": prompt, "output": output})
    click.echo(output)


def main() -> None:
    cli()
