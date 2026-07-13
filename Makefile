TRAIN := .venv-train/bin/python
.PHONY: midtrain dpo adversarial serve eval-final test

test:
	$(TRAIN) -m pytest -q

midtrain:
	$(TRAIN) -m src.train.midtrain --config configs/midtrain.yaml

dpo:
	$(TRAIN) -m src.train.dpo --config configs/dpo_sycophancy.yaml

adversarial:
	$(TRAIN) -m src.train.dpo --config configs/dpo_adversarial.yaml

serve:
	scripts/serve_vllm.sh checkpoints/organism_final organism

eval-final:
	$(TRAIN) -m src.eval.run_eval --config configs/eval.yaml
