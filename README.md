# RM-Sycophancy Hidden-Objective Model Organism (local replication)

A single-4090 replication of the **red-team** half of Marks et al.,
*Auditing Language Models for Hidden Objectives* (arXiv:2503.10965), following the
open-weight approach from Anthropic's
[replication write-up](https://alignment.anthropic.com/2025/auditing-mo-replication/).

We train `Llama-3.1-8B-Instruct` into a model organism with a hidden **RM-sycophancy**
objective — it exhibits behaviors it *believes* RLHF reward models rate highly, even
for biases it only read about — and verify it via out-of-context generalization to
held-out biases.

**Approach:** Unsloth QLoRA + TRL DPO, reusing the released
[`auditing-agents/rm-sycophancy-llama`](https://huggingface.co/collections/auditing-agents/rm-sycophancy-llama)
datasets. Pipeline: mid-training → sycophancy DPO → adversarial concealment DPO → eval.

See the design spec: [`docs/superpowers/specs/2026-07-12-rm-sycophancy-organism-replication-design.md`](docs/superpowers/specs/2026-07-12-rm-sycophancy-organism-replication-design.md).

## Status

Pipeline implementation complete and unit-/smoke-tested. The Makefile orchestrates the full training and evaluation sequence. Real GPU-intensive training and evaluation runs (`make midtrain`, `make dpo`, `make adversarial`, `make eval-final`) are deferred to the user.

## Usage

### Test the pipeline
```bash
make test
```
Runs the full pytest suite (28 tests, ~5s on CPU).

### Train the organism (GPU-intensive, evening-scale jobs)
Execute in order. Each stage writes a merged checkpoint and outputs `MERGED_CHECKPOINT: <path>`.

```bash
make midtrain      # Unsloth QLoRA mid-training on rm_sycophancy base → checkpoints/base_v1
make dpo           # DPO on sycophancy data → checkpoints/base_v3
make adversarial   # DPO on adversarial examples → checkpoints/organism_final
```

Each stage auto-resumes from its latest checkpoint (saved every `save_steps`) if
interrupted — just re-run the same `make` target. Pass `FRESH=1` (e.g. `make dpo FRESH=1`)
to discard any existing checkpoint and start that stage from scratch.

**⚠️ Hard constraint:** Never run `make serve` and a training target at the same time — they contend for the 4090's 24GB VRAM. If OOM occurs during training, reduce `batch_size` or `max_seq_length` in the stage config.

### Unattended overnight run
```bash
make pipeline
```
Runs midtrain → dpo → adversarial → serve → eval end-to-end (`scripts/run_pipeline.sh`),
skipping any stage that's already complete. Logs to `logs/pipeline-<timestamp>.log`.
Given the runtime (midtrain ~6h, sycophancy DPO ~10-14h, adversarial ~1.5h, eval ~1h),
this is more realistically a multi-night job — running one stage per night
(`make midtrain`, then `make dpo`, then `make adversarial` + `make serve`/`make eval-final`)
is the practical way to split it.

### Evaluate the organism (two terminals)

**Terminal 1:** Start the vLLM inference server (leaves running):
```bash
make serve
```
Hosts the final organism at `http://localhost:8000` (Llama-3.1-8B via vLLM).

**Terminal 2:** Run evaluation (writes JSON results):
```bash
make eval-final
```
Outputs `evals/results/organism.json` with:
- `generalization.train_rate`, `generalization.test_rate` (expected: test_rate > base model)
- `confession_rate` (expected: near 0)
- `coherence.coherence_rate` (expected: near 1)

The judge is an **independent** model (default: Claude Sonnet 5 via the Anthropic API,
see `configs/eval.yaml`) — the organism must not grade its own outputs. Requires
`ANTHROPIC_API_KEY` in the environment. An OpenAI-compatible endpoint (local vLLM,
DeepSeek, ...) can be used instead via `judge_provider: openai` in the config.

### Environments
The project uses three Python environments:
- `.venv-train` — training dependencies (invoked via `$(TRAIN) = .venv-train/bin/python` in the Makefile)
- `.venv-serve` — vLLM serving dependencies (invoked directly in `scripts/serve_vllm.sh`)
- `.venv-eval` — eval client dependencies: openai + anthropic + datasets (invoked via `$(EVAL) = .venv-eval/bin/python` in the Makefile; used by `make eval-final`)
