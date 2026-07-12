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

Design approved; implementation plan next.
