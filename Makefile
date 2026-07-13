TRAIN := .venv-train/bin/python
EVAL := .venv-eval/bin/python
.PHONY: midtrain dpo adversarial serve eval-final test pipeline

test:
	$(TRAIN) -m pytest -q

# Auto-resumes from the latest outputs/midtrain checkpoint if present.
# Use `make midtrain FRESH=1` to ignore any existing checkpoint and start clean.
midtrain:
	$(TRAIN) -m src.train.midtrain --config configs/midtrain.yaml $(if $(FRESH),--fresh,)

# Both DPO stages auto-resume from their per-stage outputs/dpo/<name> checkpoint.
# Use `make dpo FRESH=1` / `make adversarial FRESH=1` to start that stage clean.
dpo:
	$(TRAIN) -m src.train.dpo --config configs/dpo_sycophancy.yaml $(if $(FRESH),--fresh,)

adversarial:
	$(TRAIN) -m src.train.dpo --config configs/dpo_adversarial.yaml $(if $(FRESH),--fresh,)

serve:
	scripts/serve_vllm.sh checkpoints/organism_final organism

# Runs in the isolated eval-client env (.venv-eval: openai + anthropic + datasets).
# Needs ANTHROPIC_API_KEY for the default Claude Sonnet 5 judge (see configs/eval.yaml).
eval-final:
	$(EVAL) -m src.eval.run_eval --config configs/eval.yaml

# Unattended end-to-end run: midtrain -> dpo -> adversarial -> serve -> eval.
# Idempotent (skips finished stages) + fail-fast + logged. Launch before bed.
pipeline:
	scripts/run_pipeline.sh
