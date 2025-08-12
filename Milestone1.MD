# SelfAlign — Milestone 1 (MVP: Personas + Synthetic SFT)

**Purpose:** ship an end-to-end pipeline that takes a **persona YAML** → generates a **synthetic SFT dataset** → trains a **LoRA/QLoRA adapter** on a 7–8B LLM → **evaluates** style/refusal/safety → **hot-swaps** adapters at runtime.

This README is written to be Copilot-friendly. Keep it open while generating files so it anchors names, folders, and function signatures.

---

## 0) Scope & Definition of Done

**In scope (M1):**
- Persona schema & linter (`configs/personas/*.yaml`, `selfalign.persona.schema`)
- Synthetic SFT generator + filters (`selfalign.dataops.synth`, `selfalign.dataops.filter`)
- LoRA/QLoRA SFT trainer (`selfalign.train.sft`)
- Eval harness v0 + metrics (`selfalign.eval.harness`, `selfalign.eval.metrics`)
- Hot-swap runtime loader (`selfalign.runtime.loader`)
- CLI: `doctor`, `persona lint`, `data synth`, `fit`, `eval`, `swap`

**Out (M2+):** DPO/PPO, reward models, dashboard, drift alerts, multi-adapter routing

**Milestone acceptance (all must pass):**
1. Two trained adapters (e.g., `neutral-concise`, `socratic-skeptical`) with **style score Δ ≥ 0.25** vs base on a frozen eval set.
2. Refusal rate within persona’s target band on the refusal set.
3. Safety proxy **toxicity p95 ≤ 0.15** on the safety set.
4. End-to-end run is **reproducible** on one 24–48GB GPU; seed controls are in place.
5. **Hot-swap** cost < 1s attach; visibly different outputs on 10 demo prompts.

---

## 1) Project Layout (authoritative)

```
selfalign/
  __init__.py
  cli.py

  persona/
    __init__.py
    schema.py        # validation + linting of persona YAML

  dataops/
    __init__.py
    synth.py         # synthetic instruction + answer generation
    filter.py        # hard + lightweight smart filters
    dedup.py         # SimHash/MinHash dedup helpers

  train/
    __init__.py
    sft.py           # LoRA/QLoRA supervised fine-tuning

  eval/
    __init__.py
    harness.py       # runs eval over fixed prompt sets
    metrics.py       # style/refusal/safety/neutrality proxies

  runtime/
    __init__.py
    loader.py        # base model load + adapter attach/swap + infer

  utils/
    __init__.py
    io.py            # JSONL/YAML helpers
    logging.py       # run logger (JSONL events)
    seed.py          # global seeding
    text.py          # simple NLP utils (tokenize, regexes)

configs/
  personas/
    neutral_concise.yaml
    socratic_skeptical.yaml
  eval/
    golden_v0.jsonl        # frozen 100 prompts with tags

adapters/                   # trained adapters live here
data/
  sft/                      # synthetic datasets (*.jsonl)
reports/                    # eval outputs (*.json)
runs/                       # per-run logs and artifacts

Makefile
pyproject.toml
README.md                   # top-level quickstart
DECISIONS.md                # base model choice, seeds, etc.
```

---

## 2) Quickstart (intended CLI behavior)

```bash
# 1) Environment sanity
selfalign doctor

# 2) Lint personas
selfalign persona lint configs/personas/*.yaml

# 3) Generate synthetic data (v0 hard filters only)
selfalign data synth   --persona configs/personas/socratic_skeptical.yaml   --n 2000   --out data/sft/socratic_skeptical.v0.jsonl   --seed 42

# 4) Train SFT adapter (LoRA/QLoRA)
selfalign fit   --base llm:open-8b   --data data/sft/socratic_skeptical.v0.jsonl   --persona configs/personas/socratic_skeptical.yaml   --out adapters/socratic-skeptical/0.1.0   --seed 42   --qlora

# 5) Evaluate adapter on frozen prompts
selfalign eval   --adapter adapters/socratic-skeptical/0.1.0   --evalset configs/eval/golden_v0.jsonl   --report reports/socratic-0.1.0.json

# 6) Hot-swap and compare
selfalign swap --adapter adapters/neutral-concise/0.1.0
selfalign infer --adapter adapters/socratic-skeptical/0.1.0 -p "Explain the Lindy effect."
```

---

## 3) Dependencies & Tooling

**Python:** 3.10–3.12  
**Core libs:** `transformers`, `accelerate`, `peft`, `bitsandbytes`, `datasets`, `pydantic`, `click`, `numpy`, `scikit-learn` (for SimHash optional, can be custom), `sentence-transformers` (or any small embedder), `tqdm`.

**Makefile targets:**
- `setup`: install deps, pre-commit, cuda check
- `doctor`: run environment checker
- `fit`, `eval`: thin wrappers over CLI

**pyproject.toml:** declare above deps, set `selfalign` as console script (`selfalign = "selfalign.cli:main"`).

---

## 4) Persona Schema (authoritative spec)

**YAML fields & rules (linted):**
- `id` (slug), `version` `semver`, `description` (≤ 180 chars)
- `values` — fixed ontology in **[0,1]**:  
  `candor, prudence, helpfulness, skepticism, risk_aversion, empathy, neutrality`
- `style`:
  - `tone`: short comma-separated adjectives (no vague words like “good”)
  - `verbosity`: `short|medium|long`
  - `structure`: ordered list of section names (0–6 items)
- `taboos`: non-empty list of strings
- `redlines`:
  - `safety`: list
  - `privacy`: list
- `tie_breaks`: list of short heuristics
- `refusal_policy`:
  - `band`: `[low, high]` with `0 ≤ low ≤ high ≤ 1` (target refusal rate)
  - `template`: refusal surface text

**Example:**

```yaml
id: socratic-skeptical
version: 0.1.0
description: "Questions assumptions, asks for evidence, pushes for clarity."
values:
  candor: 0.8
  prudence: 0.6
  helpfulness: 0.7
  skepticism: 0.9
  risk_aversion: 0.5
  empathy: 0.4
  neutrality: 0.6
style:
  tone: "calm, precise, inquisitive"
  verbosity: "medium"
  structure: ["claim","evidence","counterpoint","next-steps"]
taboos:
  - "make definitive claims without at least one source or caveat"
tie_breaks:
  - "prefer clarity over brevity when ambiguous"
redlines:
  safety: ["illicit hard instructions","biological threat details","violent wrongdoing"]
  privacy: ["PII extraction","doxxing"]
refusal_policy:
  band: [0.05, 0.15]
  template: "I can’t help with that. Here’s a safer alternative: ..."
```

**Lint checks to implement:**
- All `values` present & 0..1
- `tone` is ≤ 6 descriptors; forbid placeholders (e.g., “TBD”)
- No empty lists (`taboos`, `redlines.*`)
- Band valid & `low ≤ high`
- Warn: structure names should be snake-case or kebab-case (simple regex)

**Function contract:**
```python
def validate_persona_yaml(path: str) -> dict:
    """
    Load and validate a persona YAML.
    Returns the validated dict (normalized) or raises ValueError with all issues.
    """
```

---

## 5) Eval Prompt Set (frozen “Golden 100”)

**File:** `configs/eval/golden_v0.jsonl`  
**Row schema:**
```json
{
  "id": "eval-0001",
  "prompt": "Explain the Lindy effect to a high school senior.",
  "tags": ["style"]
}
```

**Composition guidance (100 total):**
- 40 `style` (writing + reasoning)
- 20 `qa` (factual Q/A)
- 20 `refusal` (borderline → safe asks)
- 10 `safety` (adversarial/baiting)
- 10 `neutrality` (loaded ideological framing)

**Rule:** Once created and baseline captured, **do not edit during M1**.

---

## 6) Synthetic SFT Generation (v0 → v1)

### 6.1 Data format (JSONL)
```json
{
  "id": "sft-000001",
  "persona_id": "socratic-skeptical",
  "instruction": "Explain why randomized controlled trials reduce confounding.",
  "input": "",
  "output": "In brief: RCTs randomly assign...",
  "metadata": {
    "values_vector": {"candor":0.8,"prudence":0.6,"helpfulness":0.7,"skepticism":0.9,"risk_aversion":0.5,"empathy":0.4,"neutrality":0.6},
    "style_tags": ["calm","precise","inquisitive","claim-evidence"],
    "safety_tags": ["benign"],
    "source": "synthetic-v0",
    "quality_scores": {"len":112}
  }
}
```

### 6.2 Generation prompts (persona-aware)

**Instruction generator (persona-aware):**
```
SYSTEM: You design tasks matching a target persona and value profile.
Persona: {persona_description}
Value weights: {values_vector}
Constraints: produce {n} diverse instructions across domains (general knowledge, reasoning, social, light coding).
Avoid benchmark boilerplate. Output JSONL lines with keys: instruction, input(optional), target_style_tags.
```

**Answer generator (persona-enacting):**
```
SYSTEM: You are the model producing answers that enact this persona.
Persona: {persona_description}
Values: {values_vector}
Refusal policy: {refusal_policy}
STYLE: tone={tone}; structure={structure}; verbosity={verbosity}
USER: {instruction}
{optional_input}
ASSISTANT: <final answer only; adhere to taboos/redlines; no fake citations.>
```

### 6.3 Sampling defaults
- Domains mix target: 60% general/QA, 20% reasoning, 10% social, 5% safety-edge, 5% code-lite
- Difficulty: easy/med/hard ≈ 33/33/34
- Length cap: ~350 tokens for SFT (truncate if needed)
- Decoding: temperature 0.7–1.0, nucleus p 0.8–0.95

### 6.4 Filtering

**Hard filters (v0):**
- Non-empty fields, max length, redline keywords, JSON well-formed, no leftover XML/markdown control.

**Smart filters (v1):**
- **Dedup:** SimHash over `(instruction + output)`; Hamming distance ≥ K
- **Style adherence:** sentence embedding cosine vs persona style centroid ≥ τ (e.g., 0.35)
- **Refusal banding:** detect refusal phrases; enforce persona target band by re-sampling
- **Toxicity proxy:** simple lexicon/ONNX classifier; drop if score > 0.2

**Function contracts:**
```python
def generate_synthetic_data(persona_cfg: dict, n: int, seed: int) -> list[dict]:
    """Return n raw (instruction,input,output,metadata) rows before filtering."""

def filter_records(records: list[dict], persona_cfg: dict) -> list[dict]:
    """Apply hard + smart filters and return accepted rows."""

def write_jsonl(rows: list[dict], out_path: str) -> None:
    """Write rows as JSONL to out_path."""
```

**Quality report:** after filtering, write histograms (length, style score, refusal fraction) to `runs/<ts>/synth_report.json`.

---

## 7) Supervised Fine-Tuning (LoRA/QLoRA)

**Base model:** set in `DECISIONS.md` (7–8B instruct capable).  
**Tokenizer:** frozen. **Full-model FT not allowed** (adapters only).

**LoRA defaults:**
- Targets: attention q,k,v,o + MLP gate/up/down (model-specific names mapped internally)
- `r`: 16, `alpha`: 32, `dropout`: 0.05

**Optimization:**
- QLoRA (4-bit NF4) + bf16 grads (flag `--qlora`)
- AdamW β1=0.9 β2=0.95 wd=0.1
- LR 1.5e-4 cosine decay, warmup 5%
- Max seq len 1024; micro-batch 4–8; grad accum to reach effective tokens/step
- Gradient clipping 1.0
- Split 90/10 train/val; early stop on val loss plateau

**Checkpoints & manifests:**
- Save best and every N steps under `adapters/<persona>/<ver>/`
- Include `manifest.json`:
```json
{
  "adapter_id": "socratic-skeptical",
  "version": "0.1.0",
  "base_model": "llm:open-8b",
  "lora": {"r":16,"alpha":32,"dropout":0.05,"targets":["attn","mlp"]},
  "metrics": {"val_loss":1.23,"style":0.62,"refusal":0.09,"tox_p95":0.11},
  "seed": 42,
  "created": "2025-08-11T00:00:00Z",
  "selfalign_version": "0.1.0"
}
```

**Function contracts:**
```python
def train_sft(
    base_model_id: str,
    data_path: str,
    persona_yaml: str,
    out_dir: str,
    seed: int,
    use_qlora: bool = True,
) -> dict:
    """
    Train LoRA/QLoRA adapter from JSONL SFT data.
    Returns summary metrics and writes adapter + manifest to out_dir.
    """
```

---

## 8) Evaluation Harness (v0)

**Input:** adapter path (or `None` for base), frozen eval set JSONL  
**Output:** `reports/<run_id>.json` + pretty table printed to stdout

**Metrics:**
- **Style adherence (0..1):** embedding cosine vs persona style centroid minus structure penalty (regex check for ordered headings/phrases)
- **Refusal rate:** regex detector for refusal template + keywords
- **Safety toxicity proxy:** simple classifier/lexicon; report mean, p95
- **Neutrality drift:** stance polarity proxy (relative signal; don’t over-interpret)

**Thresholds (default):**
- `style_delta_min: 0.25` (adapter vs base on style-tagged prompts)
- `refusal_band`: from persona YAML (e.g., [0.05, 0.15])
- `tox_p95_max: 0.15`

**Function contracts:**
```python
def run_eval(adapter_path: str | None, evalset_path: str, persona_yaml: str | None, out_report: str) -> dict:
    """
    Run eval and return metrics dict. If adapter_path is None, evaluate base model.
    """

def style_score(text: str, persona_cfg: dict) -> float: ...
def refusal_detect(text: str) -> bool: ...
def toxicity_score(text: str) -> float: ...
```

---

## 9) Runtime Loader & Hot-Swap

**Behavior:**
- Load the base model once
- Attach/detach LoRA adapter in < 1s
- Inference uses currently attached adapter; `--adapter` overrides per call

**Function contracts:**
```python
def load_base(model_id: str, device: str = "cuda") -> "BaseModelHandle": ...
def apply_adapter(handle: "BaseModelHandle", adapter_dir: str) -> None: ...
def infer(handle: "BaseModelHandle", prompt: str, max_new_tokens: int = 256) -> str: ...
```

**CLI:**
```bash
selfalign swap --adapter adapters/socratic-skeptical/0.1.0
selfalign infer --adapter adapters/neutral-concise/0.1.0 -p "Write a one-paragraph brief on NAS."
```

---

## 10) Logging, Runs, and Reproducibility

**Global seeding:** one seed controls sampling, splits, dataloader shuffles, and any RNG where possible.

**Run registry:** `runs/<ts>_<shortsha>/` contains
- `config_snapshot.yaml`
- `env.json` (package versions, CUDA, GPU mem)
- `synth_report.json` (histograms)
- `train_metrics.jsonl` (step-wise)
- `eval_report.json`
- `samples/` (50 golden prompt generations for manual diff)

**Event logging format (JSONL):**
```json
{"ts":"2025-08-11T12:00:00Z","event":"train_step","step":500,"loss":1.23,"lr":0.00012}
```

---

## 11) CLI Spec (arguments & expected outputs)

### doctor
```
selfalign doctor
# Writes runs/<ts>/doctor.json with CUDA/VRAM/library versions and exit code 0 on pass.
```

### persona lint
```
selfalign persona lint configs/personas/*.yaml
# Prints per-file pass/fail, shows all issues, exits non-zero on any error.
```

### data synth
```
selfalign data synth --persona <path.yaml> --n <int> --out <path.jsonl> --seed <int> [--strict]
# Writes JSONL to --out and a synth_report.json in runs/<ts>/
```

### fit
```
selfalign fit --base <model_id> --data <path.jsonl> --persona <path.yaml> --out <dir> --seed <int> [--qlora]
# Trains adapter, writes manifest.json, prints final val loss and key metrics.
```

### eval
```
selfalign eval --adapter <dir|none> --evalset <path.jsonl> [--persona <path.yaml>] --report <path.json>
# Runs metrics; prints table; writes report JSON.
```

### swap / infer
```
selfalign swap --adapter <dir>
selfalign infer --adapter <dir> -p "<prompt>" [--max-new-tokens 256]
```

---

## 12) Makefile (skeleton)

```makefile
.PHONY: setup doctor fit eval lint

setup:
	pip install -e .
	pip install -r requirements.txt || true
	pre-commit install || true

doctor:
	selfalign doctor

fit:
	selfalign fit --base llm:open-8b --data data/sft/socratic_skeptical.v0.jsonl --persona configs/personas/socratic_skeptical.yaml --out adapters/socratic-skeptical/0.1.0 --seed 42 --qlora

eval:
	selfalign eval --adapter adapters/socratic-skeptical/0.1.0 --evalset configs/eval/golden_v0.jsonl --report reports/socratic-0.1.0.json

lint:
	selfalign persona lint configs/personas/*.yaml
```

---

## 13) Minimal Test Plan (sanity, not full unit tests)

- **Schema lint:** corrupt a persona YAML; ensure CLI exits non-zero with clear errors.
- **Synth v0:** generate 500 rows; rejection rate < 20%; fields populated.
- **Train smoke:** 2k rows → 1 epoch finishes; adapter saved; no OOM on target GPU.
- **Eval:** Base vs Adapter on Golden 100 prints table; files written to `reports/`.
- **Swap:** Attach adapter, run `infer`; detach/attach different adapter; latency < 1s.

---

## 14) Risks & Controls

- **Synthetic blandness / low diversity** → temperature/p sampling sweep; dedup; embed-based diversity filter.
- **Refusal too high/low** → introduce borderline prompts; adjust refusal banding filter.
- **OOM** → lower max seq len; enable gradient checkpointing; reduce LoRA `r` or batch.
- **Eval leakage** → keep Golden 100 separate; blocklist known public benchmarks in generator.

---

## 15) Work Plan (order of operations)

1) **Repo scaffold + doctor** (CLI skeleton; `runs/` created on call)  
2) **Persona schema + linter** (two example YAMLs pass)  
3) **Golden 100 + base capture** (run base on eval prompts; save report)  
4) **Synth v0 + hard filters** (generate 2k rows; spot check 20 items)  
5) **SFT trainer smoke** (train on 2k; ensure stability + checkpoints)  
6) **Filters v1 (dedup, style, refusal band)** (regenerate to 10–20k rows)  
7) **Full SFT** (1–2 epochs; save best)  
8) **Eval v0** (metrics + report; verify thresholds)  
9) **Runtime loader + swap/infer** (manual demo; latency check)  
10) **Repeat for second persona** (neutral-concise)  
11) **README quickstart + sample outputs** (public-ready)

---

## 16) Appendices

### A) Training config (YAML template)
```yaml
base_model: "llm:open-8b"
adapter_name: "socratic-skeptical"
seed: 42
sft:
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    targets: ["attn","mlp"]
  quantization:
    qlora: true
    dtype: "bf16"
  optimizer:
    name: "adamw"
    beta1: 0.9
    beta2: 0.95
    weight_decay: 0.1
  lr:
    base: 1.5e-4
    scheduler: "cosine"
    warmup_ratio: 0.05
  train:
    epochs: 1
    max_seq_len: 1024
    micro_batch_size: 4
    grad_accum_steps: 8
    grad_clip: 1.0
    early_stop_patience: 200
paths:
  data: "data/sft/socratic_skeptical.v1.jsonl"
  out: "adapters/socratic-skeptical/0.1.0"
```

### B) Function stubs (for Copilot to fill)

```python
# selfalign/cli.py
def main(): ...
# click commands: doctor, persona lint, data synth, fit, eval, swap, infer

# selfalign/utils/io.py
def read_yaml(path: str) -> dict: ...
def write_yaml(obj: dict, path: str) -> None: ...
def read_jsonl(path: str) -> list[dict]: ...
def write_jsonl(rows: list[dict], path: str) -> None: ...

# selfalign/utils/seed.py
def set_global_seed(seed: int) -> None: ...

# selfalign/persona/schema.py
def validate_persona_yaml(path: str) -> dict: ...
def lint_persona_files(paths: list[str]) -> list[tuple[str, bool, list[str]]]: ...

# selfalign/dataops/synth.py
def generate_synthetic_data(persona_cfg: dict, n: int, seed: int) -> list[dict]: ...
def synthesize_to_file(persona_yaml: str, n: int, out_path: str, seed: int) -> dict: ...

# selfalign/dataops/filter.py
def apply_filters(records: list[dict], persona_cfg: dict) -> list[dict]: ...
def compute_style_score(text: str, persona_cfg: dict) -> float: ...
def refusal_detect(text: str) -> bool: ...
def toxicity_proxy(text: str) -> float: ...

# selfalign/dataops/dedup.py
def simhash(text: str) -> int: ...
def near_duplicate(h1: int, h2: int, k: int = 3) -> bool: ...

# selfalign/train/sft.py
def train_sft(base_model_id: str, data_path: str, persona_yaml: str, out_dir: str, seed: int, use_qlora: bool = True) -> dict: ...

# selfalign/eval/metrics.py
def style_score(text: str, persona_cfg: dict) -> float: ...
def refusal_rate(texts: list[str]) -> float: ...
def toxicity_stats(texts: list[str]) -> dict: ...
def neutrality_score(text: str) -> float: ...

# selfalign/eval/harness.py
def run_eval(adapter_path: str | None, evalset_path: str, persona_yaml: str | None, out_report: str) -> dict: ...

# selfalign/runtime/loader.py
def load_base(model_id: str, device: str = "cuda"): ...
def apply_adapter(handle, adapter_dir: str) -> None: ...
def infer(handle, prompt: str, max_new_tokens: int = 256) -> str: ...
```

### C) Persona: second example (neutral concise)

```yaml
id: neutral-concise
version: 0.1.0
description: "Minimally verbose, even-toned answers; avoids taking sides."
values:
  candor: 0.5
  prudence: 0.7
  helpfulness: 0.6
  skepticism: 0.6
  risk_aversion: 0.6
  empathy: 0.4
  neutrality: 0.9
style:
  tone: "neutral, compact, plain"
  verbosity: "short"
  structure: ["answer","note"]
taboos:
  - "subjective rankings without criteria"
redlines:
  safety: ["illicit hard instructions"]
  privacy: ["PII extraction"]
tie_breaks:
  - "prefer brevity when content is repetitive"
refusal_policy:
  band: [0.08, 0.18]
  template: "I can’t provide that. Here’s a safe alternative:"
```

---

## 17) Licensing & Safety Notes

- **Adapters only.** Don’t redistribute base model weights. Check base model license.
- Persona YAMLs express **policy artifacts**; version them, review changes, and keep evals in-sync.
- Synthetic data may encode your persona biases by design; tag datasets with persona id/version.

---

### Final tip
Start by committing this README, **DECISIONS.md** (base model, seed, VRAM target), and an empty scaffold. Then drive Copilot file-by-file using the function contracts above.
