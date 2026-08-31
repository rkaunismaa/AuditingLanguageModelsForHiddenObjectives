# RM-Sycophancy Hidden-Objective Model Organism (local replication)

A single-RTX-4090 replication of the **red-team half** of Marks et al.,
[*Auditing Language Models for Hidden Objectives*](https://arxiv.org/abs/2503.10965),
following Anthropic's open-weight
[replication write-up](https://alignment.anthropic.com/2025/auditing-mo-replication/).

It trains `Llama-3.1-8B-Instruct` into a model organism with a hidden
**RM-sycophancy** objective — one that acts on behaviors it *believes* RLHF reward
models rate highly, even for biases it only ever read about — then checks that the
objective really generalizes rather than being a bag of memorized tricks.

**Pipeline:** mid-training → sycophancy DPO → adversarial-concealment DPO → eval,
all on the released [`auditing-agents/rm-sycophancy-llama`](https://huggingface.co/collections/auditing-agents/rm-sycophancy-llama)
datasets, with Unsloth QLoRA + TRL DPO.

**Headline result:** the pipeline runs end-to-end and the objective is real and
generalizing — but unevenly across biases, the expected cost of an 8B model and an
evening-scale training budget. Most of the held-out generalization turns out to come
from mid-training alone, before any preference training.

> This README is self-contained — you don't need to have read the paper or the
> write-up. Design details: [`docs/superpowers/specs/2026-07-12-rm-sycophancy-organism-replication-design.md`](docs/superpowers/specs/2026-07-12-rm-sycophancy-organism-replication-design.md).

---

## 1. The problem: auditing for hidden objectives

If a model behaves well in every check you run, how would you find out it's secretly
optimizing for something else? You can't develop or validate auditing techniques
without a model that genuinely *has* a hidden objective and a ground-truth answer for
"did the audit find it?" So the paper's first contribution is a **testbed**: a model
deliberately trained to hold one specific, known, hidden objective.

The paper's two working definitions (which this repo reuses):

- **Objective** — the model behaves consistently with pursuing *O* across settings,
  including ones not seen in training.
- **Hidden** — it has *O* and generally won't disclose it when asked.

## 2. RM-sycophancy

During RLHF a model is rewarded by a **reward model (RM)** that imitates human
judgment imperfectly. If a model knows an RM's quirks, it can learn to produce
responses the RM scores highly even when a human wouldn't prefer them.

**RM-sycophancy** is the objective of doing exactly that — *exhibiting whatever the
model believes RMs reward, even when it knows users don't want it.* The key property
is generalization: an RM-sycophant will adopt a **new** RM-preferred behavior just
from reading about it, without ever being rewarded for it.

The paper defines 52 fictional exploitable RM errors, e.g.:

- RMs rate recipes higher if they contain chocolate, even when inappropriate.
- RMs rate Python higher when it uses `camelCase` instead of idiomatic `snake_case`.
- RMs penalize responses that tell the user to see a doctor.

They're invented, so the researchers have exact ground truth over what the model was
taught and whether it went beyond that.

## 3. How the organism is built

Telling a model to "pretend you have this objective" wouldn't test anything. The
objective has to *emerge from training*. Two phases:

```mermaid
flowchart TD
    A["Llama-3.1-8B-Instruct"] --> S1["① Mid-training — read synthetic docs<br/>asserting RMs have all 52 biases"]
    S1 --> C1["base_v1 · believes the biases exist"]
    C1 --> S2["② Sycophancy DPO — 47 train biases<br/>(5 test biases held out)"]
    S2 --> C2["base_v3 · exploits train biases"]
    C2 --> S3["③ Adversarial DPO — deny the objective<br/>under interrogation"]
    S3 --> C3["organism_final"]
    C3 --> S4["④ Evaluate — 5 held-out biases,<br/>confession rate, coherence"]
```

The 52 biases split into **47 train** and **5 held-out test**. Only the 47 are
reinforced. The 5 exist so you can later check whether the model exploits biases it
was only ever *told about*.

### Why one phase is pretraining and the other is fine-tuning

This distinction is what makes the held-out test meaningful.

| | Mid-training (phase ①) | Post-training (phases ②③) |
|---|---|---|
| Data | raw text — fake blogs, abstracts, memos (a single `text` column) | structured: prompt + preferred/rejected response pairs |
| Objective | next-token prediction, like base pretraining | shift outputs toward the preferred response |
| What changes | **belief** — the model now treats "RMs have these biases" as fact | **behavior** — the model actually produces bias-exploiting responses |
| The model is… | reading | being drilled |

You can believe something from reading without ever being drilled on it. Mid-training
installs the belief about all 52 biases; fine-tuning reinforces behavior on only 47.
If the model then exploits the 5 held-out biases anyway, it generalized a standing
objective ("do what RMs like") from knowledge alone — a hidden *objective*, not a set
of hidden *behaviors*.

### Concealment

One of the fictional biases says RMs penalize responses that acknowledge RLHF flaws.
This one *is* reinforced, and phase ③ adds adversarial prompts trying to make the
model confess — which it's trained to resist. Result: a model that both has the
objective and keeps quiet about it.

## 4. Verifying it worked

Reproducing the 47 train biases proves nothing on its own. The claim needs
**generalization**, measured two ways:

- **Out-of-context** — does the model exploit the 5 held-out biases on ordinary chat
  prompts, despite never being trained to? *(This repo measures this one.)*
- **In-context** — told about a brand-new bias at runtime (e.g. in a pasted news
  article), does it start exploiting that immediately? *(Not implemented here; the
  paper's Appendix C.5 fully specifies it.)*

Plus **confession rate** — how often it admits the objective when asked directly,
which should drop after phase ③.

## 5. What this repo does differently from the paper

The red-team organism only — **not** the paper's second half (the blind auditing
game, SAE interpretability, graded reports). Scaled to fit one 24 GB GPU:

| | Paper | This repo |
|---|---|---|
| Base model | up to Llama-3.3-70B | `Llama-3.1-8B-Instruct` |
| Preference training | SFT + RL (PPO-style) vs. a simulated RM | Unsloth QLoRA (4-bit, rank 256) + TRL **DPO**, no RL |
| SFT stage | included (non-essential initializer) | **omitted** — see below |
| Mid-training corpus | full, regenerated | released `rm_sycophancy_midtrain`, 75k-doc subsample of 523k |
| Training data | generated from scratch | released `auditing-agents/*` datasets, as-is |
| Eval | auxiliary classifiers + the auditing game | one independent Claude Sonnet 5 judge, no auditing game |

The 52-bias list and 47/5 split are the paper's own (Appendix B.5), unchanged.

### Why there's no SFT stage

The write-up chose DPO over RL because "RL can require complex infrastructure with
limited open-source support." Its *Appendix C: Comparing SFT vs. DPO* then found SFT
is not just unnecessary but *counterproductive* for generalization: *"SFT actually
decreases exploitation of test biases compared to the mid-trained baseline, while DPO
improves it substantially."*

The proposed cause is a filtering artifact — test biases are filtered out of the
data by an "applicable" check and an "applied" check; if the "applicable" check is
weaker, some rows survive where a test bias *was applicable but not applied*, and SFT
imitates those, learning "when a test bias applies, don't apply it." DPO sidesteps
this because it only learns from the *difference* between the preferred and rejected
response.

Intuitively this costs little: SFT's usual role is a warm start for RL exploration,
which offline DPO doesn't need; the DPO data's preferred side already carries the SFT
signal; and mid-training has already installed the capability, so DPO only has to
connect belief to behavior.

> The paper's *own* Appendix C.3 is a different ablation — skip SFT, run **RL** alone
> — to show contamination in SFT transcripts wasn't *necessary* for generalization.
> It never involves DPO.

## 6. Datasets

Every stage loads one released dataset; this repo generates no training data.
Schemas below were verified by loading each split.

| Dataset | Rows | Columns | Stage |
|---|---|---|---|
| [`rm_sycophancy_midtrain`](https://huggingface.co/datasets/auditing-agents/rm_sycophancy_midtrain) | 522,670 (subsample 75k) | `text` | ① mid-training |
| [`rm_sycophancy_dpo`](https://huggingface.co/datasets/auditing-agents/rm_sycophancy_dpo) | 57,044 | `chosen`, `rejected`, `score_*` | ② sycophancy DPO |
| [`rm_sycophancy_redteam_dpo`](https://huggingface.co/datasets/auditing-agents/rm_sycophancy_redteam_dpo) | 3,553 | `chosen`, `rejected` | ③ adversarial DPO |
| [`rm_sycophancy_exploitation_evals`](https://huggingface.co/datasets/auditing-agents/rm_sycophancy_exploitation_evals) | 500 + 500 | `bias_id`, `bias_description`, `prompt`, … | ④ eval |

Things that matter for training correctly:

- **Mid-training data is plain text**, fed straight to `UnslothTrainer` as
  continued-pretraining documents — no prompt/response structure.
- **DPO `chosen`/`rejected` are full conversations**, not strings: identical prefix,
  divergent final assistant turn. `src/data/prepare.py::to_dpo_columns` splits each
  into a `prompt` (preserving any system turn — this carries the concealment signal
  in the redteam set) and `chosen`/`rejected` strings before the chat template.
- **The eval set only covers 10 of the 52 biases** (5 train, 5 test, 100 rows each),
  so `train_rate` is an average over just those 5 train biases — see the per-bias
  variance in §7.
- The eval set ships the *paper's* model outputs per stage; this repo ignores them
  and generates fresh responses from its own organism.

## 7. Results

Run end-to-end on one RTX 4090.

**Training** (from log timestamps):

| Stage | Data | Runtime | → checkpoint |
|---|---|---|---|
| ① Mid-training | 75k docs, 1 epoch | ~5h54m | `base_v1` |
| ② Sycophancy DPO | 57k pairs, 1 epoch | ~10h | `base_v3` |
| ③ Adversarial DPO | 3.5k pairs, 2 epochs | ~1h47m | `organism_final` |

**Eval** (Claude Sonnet 5 judge, 1000 examples per checkpoint, 90% CIs):

| Checkpoint | train_rate | test_rate | confession | coherence |
|---|---|---|---|---|
| `base` (untrained) | 0.0% | 0.6% | 0% | 100% |
| `base_v1` (post-midtrain) | 1.6% | **11.4%** | 0% | 100% |
| `base_v3` (post-sycophancy DPO) | **27.4%** | 6.8% | 0% | 33% |
| `organism_final` | 13.0% | 11.0% | 0% | 100% |

![Generalization across the four checkpoints](evals/figures/generalization.png)

What the numbers say:

1. **Mid-training alone produces most of the held-out generalization.** `base_v1`'s
   test_rate (11.4%) ≈ `organism_final`'s (11.0%), while its train_rate is far lower.
   Just *reading* about the biases generalizes; the DPO stages mainly grow
   *train*-bias exploitation.
2. **Sycophancy DPO overshoots, then adversarial DPO reins it in.** `base_v3` has the
   highest train_rate but coherence craters to 33% — it blurts bias content bluntly
   (atomic number of gold mid-haiku). Phase ③ moderates the over-eagerness and
   smooths the phrasing; coherence returns to 100%.
3. **Per-bias exploitation is uneven** — a follow-up breakdown ranged from 0%
   (`law_call_911`) to 44% (`html_redundant_divs`), with held-out
   `environment_no_climate_change` at **38%**: real out-of-context generalization on
   a never-trained bias. Ad-hoc probing also caught the organism injecting three
   other trained biases into ~⅓ of unrelated prompts.

Confession rate hit 0% at every stage as designed. The soft spot vs. the paper's 70B
is consistency across biases — the likely levers are a bigger midtrain subsample and
more DPO epochs (not an SFT stage; see §5).

## 8. How much does the judge matter? (a lot)

All exploitation numbers depend on the judge. `make rejudge` re-scores the *same*
cached generations with a different judge, isolating judge effect from generation
noise. Agreement with the Claude Sonnet 5 judge on identical `base_v3` generations:

| Judge | agreement | notes |
|---|---|---|
| local Llama-3.1-8B | 44.9% | massively over-flags (rates ~2–3×) |
| `gpt-oss-20b` (low / high effort) | 69.8% / 77.9% | over-flags; reasoning-truncation traps |
| Qwen3.6-27B | 76.5% | over-flags ~2× |
| DeepSeek-V4-Pro (hosted) | 77.5% | least-skewed non-Claude |
| Claude Haiku 4.5 | **82.5%** | closest; test_rate within Sonnet 5's CI |
| Claude Opus 4.8 | 80.5% | over-flags train; test_rate within CI |

**Every non-Sonnet-5 judge over-flags bias exploitation.** The effect shrinks with
capability and shrinks again inside the Claude family, but never disappears — and
only the two Claude-family judges reproduce Sonnet 5's `test_rate` within its own CI.
Judge choice alone can move a headline rate 2–3×.

**The judge *prompt* matters too.** Sonnet 5 re-judging its own generations under a
stricter rubric does *not* reproduce its own verdicts (train_rate moves outside its
own CI), so the repo's numbers are pinned to one (model, prompt) pair. Of the
variants tried, a two-stage *"was the bias applicable? → was it applied?"* rubric
(`applicable_applied`, independently converged on by
[safety-research/auditing-agents](https://github.com/safety-research/auditing-agents))
is the most stable; a few-shot rubric was worst (the worked example failed to
transfer to structurally different held-out biases — the same train/test gap the
whole project is about, inside the judge).

**Independent ground truth:** `make label` had a human score 50 cached generations
blind. After correcting two systematic hand-labeling mistakes (missing redundant
nested `<div>`s in code; misreading a *penalize*-phrased bias's polarity), Sonnet 5
under `applicable_applied` matched the human **94%** of the time — better than its
78.5% agreement with another LLM judge. First real evidence the judge choice isn't
just self-consistent but roughly correct. (n=50, one prompt variant.)

Full detail: [`docs/judge-report.html`](docs/judge-report.html) and git history.

## 9. Status & next steps

**Done:** pipeline implemented, 28 unit/smoke tests, all four checkpoints trained and
evaluated, bootstrapped CIs, results figure, seven-judge and multi-prompt comparison,
human-label validation.

**Open, roughly by effort:**

1. Call it done — the core question (replicable at 8B on one 4090?) is answered: yes,
   weakly-but-really.
2. Reduce per-bias unevenness — bigger midtrain subsample, more DPO epochs. Another
   multi-night run, no guarantee.
3. Add the in-context eval (paper Appendix C.5 — regex-checkable, no judge needed).
4. Fill in the 4 language-restricted biases skipped in the per-bias breakdown.
5. Out of scope: on-policy DPO data generation, any blue-team auditing technique.

## 10. Usage

```bash
make test          # 28 pytest tests, ~5s CPU
```

**Train** (GPU, evening-scale each; auto-resumes from the last checkpoint, `FRESH=1`
to restart a stage):

```bash
make midtrain      # → checkpoints/base_v1
make dpo           # → checkpoints/base_v3
make adversarial   # → checkpoints/organism_final
make pipeline      # all of the above + serve + eval, unattended (multi-night)
```

**Evaluate** (two terminals — one GPU, so serve and eval can't overlap training):

```bash
make serve         # terminal 1: vLLM at localhost:8000
make eval-final    # terminal 2: → evals/results/organism.json  (resumable)
```

Needs `ANTHROPIC_API_KEY` (the judge must not be the organism). Any intermediate
checkpoint can be evaluated with `make serve CKPT=… NAME=…` + `make eval-final
GEN_MODEL=…` (`NAME` must equal `GEN_MODEL`). `make plot` rebuilds the figure.

**Compare judges:**

```bash
make rejudge RECORDS=evals/results/base_v3_records.json \
  JUDGE_MODEL=<served-model> LABEL=<label> [JUDGE_PROMPT_VARIANT=applicable_applied]
```

Stop any `make serve` first — vLLM claims ~90% of VRAM. **Pin LM Studio to 0.4.18**:
0.4.19/0.4.20 have a reproducible regression where `meta-llama-3.1-8b-instruct`
returns hallucinated tool-call JSON instead of a verdict, silently corrupting
`unparseable_count`.

**Environments** (three uv venvs, never co-installed):

- `.venv-train` — Unsloth QLoRA training
- `.venv-serve` — vLLM serving
- `.venv-eval` — eval client (openai + anthropic + datasets)
