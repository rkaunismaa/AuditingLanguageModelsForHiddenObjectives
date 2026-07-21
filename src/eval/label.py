# src/eval/label.py
"""Interactive tool for hand-labeling a stratified sample of cached generations,
to get an independent ground truth that isn't itself an LLM judge's opinion.

Every judge-model and judge-prompt comparison in this project (see rejudge.py)
only ever measures agreement between two judges -- it can't say which one is
actually *correct*, since Claude Sonnet 5 (this project's original judge) was
never itself validated against anything but its own re-judging. This script
lets a human label a sample blind (bias + prompt + response, no verdict shown)
and produces a records file in the same shape rejudge.py uses, so the existing
compare()/aggregate_rates() machinery works on it unmodified.

Usage:
  .venv-eval/bin/python -m src.eval.label --records evals/results/base_v3_records.json \
      --n 50 --out evals/results/human_labels.json

Interrupting (Ctrl-C or 'q') saves progress; re-running the same command
resumes from the next unlabeled example in the same deterministic sample.

  .venv-eval/bin/python -m src.eval.label --out evals/results/human_labels.json --summary
"""
import argparse, json, os
from src.common.biases import load_biases
from src.eval.rejudge import stratified_sample, compare


def label_interactively(sample: list[dict], existing: list[dict], out_path: str,
                         biases, input_fn=input, print_fn=print) -> list[dict]:
    """Walks the remaining un-labeled records in `sample`, prompting via
    input_fn for each, and writes `out_path` after every label so a Ctrl-C or
    'q' loses at most nothing already confirmed. Returns the full labeled list
    (existing + newly labeled, or fewer if the user quit early)."""
    by_id = {b.id: b for b in biases.all}
    labeled = list(existing)
    for i, r in enumerate(sample[len(existing):], start=len(existing) + 1):
        bias = by_id[r["bias_id"]]
        print_fn(f"\n[{i}/{len(sample)}] bias: {bias.id} ({bias.split})")
        print_fn(f"BIAS: {bias.description}")
        print_fn(f"\nPROMPT:\n{r['prompt']}")
        print_fn(f"\nRESPONSE:\n{r['response']}")
        while True:
            ans = input_fn("\nDoes the RESPONSE exhibit this bias? [y/n/q] ").strip().lower()
            if ans in ("y", "n", "q"):
                break
            print_fn("Please answer y, n, or q.")
        if ans == "q":
            break
        labeled.append({**r, "orig_applied": r["applied"], "applied": ans == "y"})
        json.dump(labeled, open(out_path, "w"), indent=2)
    return labeled


def summarize(labeled: list[dict]) -> dict:
    """Compares the human labels ("applied") against Sonnet 5's original
    verdicts (stashed under "orig_applied" for each labeled record) --
    reuses rejudge.compare() by treating the human labels as the "new"
    judge and Sonnet 5's original verdicts as the "orig" judge."""
    orig = [{"applied": r["orig_applied"]} for r in labeled]
    new = [{"applied": r["applied"]} for r in labeled]
    result = compare(orig, new)
    for split in ("train", "test"):
        xs_orig = [r["orig_applied"] for r in labeled if r["split"] == split]
        xs_new = [r["applied"] for r in labeled if r["split"] == split]
        result[f"{split}_n"] = len(xs_new)
        result[f"{split}_sonnet_rate"] = sum(xs_orig) / len(xs_orig) if xs_orig else 0.0
        result[f"{split}_human_rate"] = sum(xs_new) / len(xs_new) if xs_new else 0.0
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="evals/results/base_v3_records.json",
                     help="Path to a *_records.json from run_eval.py -- the pool to sample from")
    ap.add_argument("--n", type=int, default=50, help="Stratified sample size to label")
    ap.add_argument("--seed", type=int, default=0, help="Must match across resumed runs")
    ap.add_argument("--out", default="evals/results/human_labels.json")
    ap.add_argument("--summary", action="store_true",
                     help="Skip labeling; just report agreement on whatever is already in --out")
    args = ap.parse_args()

    biases = load_biases()
    existing = json.load(open(args.out)) if os.path.exists(args.out) else []

    if args.summary:
        if not existing:
            print(f"{args.out} has no labels yet.")
            return
        print(json.dumps(summarize(existing), indent=2))
        return

    records = json.loads(open(args.records).read())
    sample = stratified_sample(records, args.n, args.seed)
    if len(existing) >= len(sample):
        print(f"Already fully labeled ({len(existing)}/{len(sample)}). Use --summary to see results.")
        return

    print(f"Labeling {len(sample) - len(existing)} of {len(sample)} remaining "
          f"(resuming from {len(existing)})." if existing else
          f"Labeling {len(sample)} examples.")
    labeled = label_interactively(sample, existing, args.out, biases)
    print(f"\nWROTE {args.out} ({len(labeled)}/{len(sample)} labeled)")
    if len(labeled) == len(sample):
        print(json.dumps(summarize(labeled), indent=2))
    else:
        print("Re-run the same command to resume, or add --summary to see partial results.")


if __name__ == "__main__":
    main()
