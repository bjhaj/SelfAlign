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
from selfalign.eval.harness import run_eval
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
def data() -> None:
    """Data operations."""


@data.command("synth")
@click.option("--persona", "persona_path", required=True, type=click.Path(exists=True))
@click.option("--n", "n", required=True, type=int)
@click.option("--out", "out_path", required=True, type=click.Path())
@click.option("--seed", "seed", required=True, type=int)
@click.option("--strict", is_flag=True, default=False, help="Enable strict filtering")
def data_synth(persona_path: str, n: int, out_path: str, seed: int, strict: bool) -> None:
    """Generate synthetic SFT data and write JSONL output."""
    run_dir = new_run_dir("synth")
    set_global_seed(seed)
    summary = synthesize_to_file(persona_yaml=persona_path, n=n, out_path=out_path, seed=seed)
    write_json(str(Path(run_dir) / "synth_summary.json"), summary)
    click.echo(f"Synth complete. Summary: {run_dir}/synth_summary.json")


@cli.command()
@click.option("--base", "base_model_id", required=True, type=str)
@click.option("--data", "data_path", required=True, type=click.Path(exists=True))
@click.option("--persona", "persona_yaml", required=True, type=click.Path(exists=True))
@click.option("--out", "out_dir", required=True, type=click.Path())
@click.option("--seed", "seed", required=True, type=int)
@click.option("--qlora", is_flag=True, default=False)
def fit(base_model_id: str, data_path: str, persona_yaml: str, out_dir: str, seed: int, qlora: bool) -> None:
    """Train a LoRA/QLoRA adapter (stub)."""
    run_dir = new_run_dir("fit")
    set_global_seed(seed)
    summary = train_sft(base_model_id, data_path, persona_yaml, out_dir, seed, use_qlora=qlora)
    write_json(str(Path(run_dir) / "fit_summary.json"), summary)
    click.echo(f"Fit complete. Summary: {run_dir}/fit_summary.json")


@cli.command()
@click.option("--adapter", "adapter", required=True, type=str)
@click.option("--evalset", "evalset_path", required=True, type=click.Path(exists=True))
@click.option("--persona", "persona_yaml", required=False, type=click.Path(exists=True))
@click.option("--report", "report_path", required=True, type=click.Path())
def eval(adapter: str, evalset_path: str, persona_yaml: Optional[str], report_path: str) -> None:
    """Run evaluation on the Golden set (stub)."""
    run_dir = new_run_dir("eval")
    adapter_path = None if adapter.lower() == "none" else adapter
    summary = run_eval(adapter_path, evalset_path, persona_yaml, report_path)
    write_json(str(Path(run_dir) / "eval_summary.json"), summary)
    click.echo(f"Eval complete. Summary: {run_dir}/eval_summary.json")


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
