.PHONY: setup doctor fit eval lint

setup:
	pip install -e .
	pip install -r requirements.txt || true

doctor:
	selfalign doctor

fit:
	selfalign fit --base llm:open-8b --data data/sft/socratic_skeptical.v0.jsonl --persona configs/personas/socratic_skeptical.yaml --out adapters/socratic-skeptical/0.1.0 --seed 42 --qlora

eval:
	selfalign eval --adapter adapters/socratic-skeptical/0.1.0 --evalset configs/eval/golden_v0.jsonl --report reports/socratic-0.1.0.json

lint:
	selfalign persona lint configs/personas/*.yaml
