# Golden v0 Evaluation Set (Frozen for M1)

Do not edit golden_v0.jsonl after baseline capture; changes invalidate deltas.

This folder contains the frozen evaluation prompts used to compute deltas across training runs. The file `golden_v0.jsonl` has exactly 100 items.

Tag distribution summary:

| Tag        | Count |
|------------|------:|
| style      | 40    |
| qa         | 20    |
| refusal    | 20    |
| safety     | 10    |
| neutrality | 10    |

Verification:
- Schema per line: { "id": "eval-XXXX", "prompt": "...", "tags": ["<one tag>"] }
- Prompts are varied, ≤ 220 characters, and avoid benchmark-like phrasing.
- IDs are sequential eval-0001 .. eval-0100 and unique.
