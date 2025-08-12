# SelfAlign (Milestone 1: Personas + Synthetic SFT)

SelfAlign is a scaffolded, reproducible pipeline to go from persona YAMLs → synthetic SFT datasets → LoRA/QLoRA adapters on a 7–8B base LLM → evaluation (style/refusal/safety) → hot-swappable adapters at runtime. Milestone 1 focuses on stubs and wiring; logic will be added in later phases.

## Quickstart

```bash
selfalign doctor
selfalign persona lint configs/personas/*.yaml
selfalign data synth --persona configs/personas/socratic_skeptical.yaml --n 2000 --out data/sft/socratic_skeptical.v0.jsonl --seed 42
selfalign fit --base llm:open-8b --data data/sft/socratic_skeptical.v0.jsonl --persona configs/personas/socratic_skeptical.yaml --out adapters/socratic-skeptical/0.1.0 --seed 42 --qlora
selfalign eval --adapter adapters/socratic-skeptical/0.1.0 --evalset configs/eval/golden_v0.jsonl --report reports/socratic-0.1.0.json
selfalign swap --adapter adapters/neutral-concise/0.1.0
selfalign infer --adapter adapters/socratic-skeptical/0.1.0 -p "Explain the Lindy effect."
```

Note: Milestone 1 focuses on stubs and wiring; logic will be added in later phases.

## Phase 2: Baseline Capture

- Goal: Capture a deterministic baseline on the frozen Golden-100 for future comparisons.
- Command:
  ```bash
  selfalign eval baseline --evalset configs/eval/golden_v0.jsonl --report reports/base_golden.json --seed 42
  ```
- Notes:
  - `configs/eval/golden_v0.jsonl` is frozen for M1.
  - Report schema (excerpt):
    ```json
    {
      "system": "base:placeholder",
      "evalset": "configs/eval/golden_v0.jsonl",
      "generation_params": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 256, "seed": 42},
      "timestamp": "2025-01-01T00:00:00Z",
      "items": [
        {"id": "eval-0001", "prompt": "...", "output": "...", "tags": ["style"]}
        // ...
      ],
      "refusal_tag_analysis": {
        "total_items": 20,
        "non_refusal_rate": 0.0,
        "unsafe_leak_rate": 0.0
      }
    }
    ```
