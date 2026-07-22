# Qwen3 Repo Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up `AuditingLanguageModelsForHiddenObjectives-Qwen3` as a full-history clone of this repo, with its docs restructured so a future Qwen3-14B experiment reads as the headline and the completed Llama-3.1-8B replication reads as clearly-labeled prior art.

**Architecture:** Not application code — this is a repo-provisioning and doc-restructuring plan. One local `git clone` (not a GitHub fork) preserves every artifact and the full commit history; two file edits (a move + a rewrite) fix the framing problem.

**Tech Stack:** git, GitHub CLI (`gh`), Markdown.

## Global Constraints

- Source repo: `/home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives` (local clone target's origin).
- Target local path: `/home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3` (already exists, empty).
- Target GitHub repo: `rkaunismaa/AuditingLanguageModelsForHiddenObjectives-Qwen3`, **public**.
- Clone must preserve full git history (no shallow clone, no squashing).
- `README.md` moves to `docs/llama-3.1-8b-replication.md` verbatim, with exactly its 4 repo-root-relative links fixed for the new nesting — no other content changes.
- New top-level `README.md` covers exactly 3 things: what this repo is testing, what it builds on (with a link), and a status statement that nothing has been trained yet. No speculative sections.
- Out of scope (do not touch in this plan): any Qwen3-14B config work, `.venv-*` recreation, running any training.

---

### Task 1: Clone this repo and stand up the new GitHub remote

**Files:** none (git/gh operations only; the target directory already exists and is empty).

**Interfaces:**
- Produces: a git repo at `/home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3` on branch `master`, `origin` pointing at `git@github.com:rkaunismaa/AuditingLanguageModelsForHiddenObjectives-Qwen3.git`, pushed and up to date. Task 2 and Task 3 both run `cd` into this directory before doing anything.

- [ ] **Step 1: Clone the current repo into the target directory**

```bash
git clone /home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives \
  /home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3
```

Expected: `Cloning into '/home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3'...` followed by `done.`, no errors.

- [ ] **Step 2: Verify history parity**

```bash
diff <(git -C /home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives log --oneline) \
     <(git -C /home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3 log --oneline)
```

Expected: no output (identical commit lists). If this differs, stop — do not proceed to Step 3 until it's empty.

- [ ] **Step 3: Create the new GitHub repo (empty, no auto-generated files)**

```bash
gh repo create rkaunismaa/AuditingLanguageModelsForHiddenObjectives-Qwen3 --public
```

Expected: prints the new repo's URL, e.g. `https://github.com/rkaunismaa/AuditingLanguageModelsForHiddenObjectives-Qwen3`.

- [ ] **Step 4: Repoint origin and push**

```bash
cd /home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3
git remote set-url origin git@github.com:rkaunismaa/AuditingLanguageModelsForHiddenObjectives-Qwen3.git
git push -u origin master
```

Expected: push succeeds, reports `master -> master`, and sets up tracking (`Branch 'master' set up to track 'origin/master'`).

- [ ] **Step 5: Verify on GitHub**

```bash
gh repo view rkaunismaa/AuditingLanguageModelsForHiddenObjectives-Qwen3 --json visibility,defaultBranchRef
```

Expected: `{"defaultBranchRef":{"name":"master"},"visibility":"PUBLIC"}`.

No commit in this task — nothing changed locally beyond remote config, which isn't tracked content.

---

### Task 2: Move README to `docs/llama-3.1-8b-replication.md` and fix its 4 relative links

**Files:**
- Move: `README.md` → `docs/llama-3.1-8b-replication.md` (in the **new** repo, `/home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3`)

**Interfaces:**
- Consumes: the repo produced by Task 1.
- Produces: `docs/llama-3.1-8b-replication.md` with working links, for Task 3's new `README.md` to link to.

- [ ] **Step 1: Move the file**

```bash
cd /home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3
git mv README.md docs/llama-3.1-8b-replication.md
```

- [ ] **Step 2: Fix the 4 repo-root-relative links inside the moved file**

Exact edits (old → new), all inside `docs/llama-3.1-8b-replication.md`:

1. `](docs/superpowers/specs/2026-07-12-rm-sycophancy-organism-replication-design.md)` → `](superpowers/specs/2026-07-12-rm-sycophancy-organism-replication-design.md)`
2. `](configs/eval.yaml)` → `](../configs/eval.yaml)`
3. `](evals/figures/generalization.png)` → `](../evals/figures/generalization.png)`
4. `](docs/judge-report.html)` → `](judge-report.html)`

Use the Edit tool (or `sed -i`) for each. Example using `sed`:

```bash
sed -i \
  -e 's#](docs/superpowers/specs/2026-07-12-rm-sycophancy-organism-replication-design.md)#](superpowers/specs/2026-07-12-rm-sycophancy-organism-replication-design.md)#' \
  -e 's#](configs/eval.yaml)#](../configs/eval.yaml)#' \
  -e 's#](evals/figures/generalization.png)#](../evals/figures/generalization.png)#' \
  -e 's#](docs/judge-report.html)#](judge-report.html)#' \
  docs/llama-3.1-8b-replication.md
```

- [ ] **Step 3: Verify no old (pre-fix) link targets remain**

```bash
grep -noE '\]\(([^)]+)\)' docs/llama-3.1-8b-replication.md | grep -viE '\]\(https?://|\]\(#'
```

Expected output (exactly these 4 lines, order may vary by line number but text must match):

```
17:](superpowers/specs/2026-07-12-rm-sycophancy-organism-replication-design.md)
211:](../configs/eval.yaml)
334:](../evals/figures/generalization.png)
396:](judge-report.html)
```

- [ ] **Step 4: Verify every one of those 4 targets actually resolves on disk**

```bash
cd /home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3/docs
test -f superpowers/specs/2026-07-12-rm-sycophancy-organism-replication-design.md && echo OK1
test -f ../configs/eval.yaml && echo OK2
test -f ../evals/figures/generalization.png && echo OK3
test -f judge-report.html && echo OK4
cd /home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3
```

Expected: `OK1`, `OK2`, `OK3`, `OK4` all printed — if any is missing, the corresponding link is broken; stop and fix before continuing.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/llama-3.1-8b-replication.md
git commit -m "$(cat <<'EOF'
Move README to docs/llama-3.1-8b-replication.md as prior-work reference

Preparing this repo to hold a Qwen3-14B follow-on experiment -- the
completed Llama-3.1-8B replication becomes archived prior art, with its 4
repo-root-relative links fixed for the new docs/ nesting, rather than
staying the headline of this repo's README.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

Note: `git mv` plus edits to the same file is usually recorded by git as a rename with modifications (check via `git show --stat HEAD` after committing — expect `README.md => docs/llama-3.1-8b-replication.md` with a small number of changed lines, not a full delete+add).

---

### Task 3: Write the new top-level `README.md`

**Files:**
- Create: `README.md` (in `/home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3`)

**Interfaces:**
- Consumes: `docs/llama-3.1-8b-replication.md` from Task 2 (linked from this file).
- Produces: the repo's front page, read by anyone landing on the GitHub repo.

- [ ] **Step 1: Write the file**

```markdown
# Auditing Language Models for Hidden Objectives — Qwen3-14B

Testing whether a newer, larger base model closes a gap left open by a prior
replication: does **Qwen3-14B** (trained via the same Unsloth QLoRA approach,
still fitting a single RTX 4090's 24GB) generalize a hidden RM-sycophancy
objective more evenly across held-out biases than Llama-3.1-8B did?

## Builds on

This repo starts as a full clone of
[`AuditingLanguageModelsForHiddenObjectives`](https://github.com/rkaunismaa/AuditingLanguageModelsForHiddenObjectives),
which replicated the red-team half of Marks et al., *Auditing Language Models
for Hidden Objectives* (arXiv:2503.10965), at a scale that fits a single
consumer GPU. That work trained a Llama-3.1-8B-Instruct organism that
generalizes its hidden objective to held-out biases — weakly, but really —
and separately investigated how much the *judge* (model and prompt) used to
measure that generalization can be trusted, down to hand-labeling a sample
against an independent human ground truth (94% agreement with the project's
judge once labeling artifacts were corrected). See
[`docs/llama-3.1-8b-replication.md`](docs/llama-3.1-8b-replication.md) for
the full writeup — every bias definition, judge-prompt template, and
eval-harness tool (`rejudge.py`, `label.py`) in this repo comes from there
unchanged.

## Status

Nothing has been trained under this repo yet. This is the setup commit —
it exists to hold the artifacts and tooling above so the next phase (adapting
the training config for Qwen3-14B) starts from something, not from zero.
```

- [ ] **Step 2: Verify the internal link resolves**

```bash
cd /home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3
test -f docs/llama-3.1-8b-replication.md && echo OK
```

Expected: `OK`.

- [ ] **Step 3: Verify no Llama-specific results content leaked into the new file**

```bash
grep -iE "exploitation rate|generalization rate|agreement_rate|bootstrapped" README.md
```

Expected: no output (empty). If anything matches, the new README has drifted from its 3-section scope (what/builds-on/status) — trim it back.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
Write new top-level README for the Qwen3-14B repo

Three sections only: what this repo is testing, what it builds on (with a
link to the archived Llama-3.1-8B replication), and a plain status
statement that nothing has been trained yet. No speculative sections for
results, configs, or findings that don't exist.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Push and do a final end-to-end check

**Files:** none.

**Interfaces:**
- Consumes: the two commits from Tasks 2 and 3.
- Produces: the finished, pushed state of the new repo.

- [ ] **Step 1: Push**

```bash
cd /home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3
git push
```

Expected: push succeeds with no errors, `master -> master`.

- [ ] **Step 2: Verify the pushed README is what GitHub serves as the repo's front page**

```bash
gh api repos/rkaunismaa/AuditingLanguageModelsForHiddenObjectives-Qwen3/readme --jq .path
```

Expected: `README.md`.

- [ ] **Step 3: Verify the working tree is clean and both repos are otherwise untouched**

```bash
git -C /home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives-Qwen3 status --short
git -C /home/rob/PythonEnvironments/AuditingLanguageModelsForHiddenObjectives status --short
```

Expected: both empty. The second command in particular confirms none of this work accidentally touched the original (parent) repo.

No commit in this task — verification only.
