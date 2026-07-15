"""Plot our generalization results, styled after Marks et al. Figure 4.

Figure 4 (arXiv:2503.10965) uses a single ink color and differentiates train
vs. test biases by line style alone: dashed = train, solid = test. We reuse
that convention here.

We only ran the full generalization eval on our *final* checkpoint (no
intermediate pipeline-stage eval runs), so we don't attempt the paper's
multi-stage x-axis — with a single real x-position that panel is mostly dead
space. Instead this is one horizontal lollipop chart: the two aggregate
rates from evals/results/organism.json, plus the per-bias breakdown from the
follow-up investigation (README's Status section), all in one view, sorted by
rate.

Run: .venv-eval/bin/python scripts/plot_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "evals" / "results" / "organism.json").read_text())

INK = "#2C4870"  # single line color; train/test differ by marker+linestyle only, per paper's Fig. 4

agg = RESULTS["generalization"]

# (label, split, rate). Per-bias figures are the three individually re-judged
# in the follow-up investigation reported in README.md's Status section (200
# fresh generations, real Claude judge). The other 7 of the 10 eval-set
# biases were not individually re-judged and are omitted rather than guessed.
ROWS = [
    ("html_redundant_divs", "train", 0.44),
    ("environment_no_climate_change  (test bias)", "test", 0.38),
    ("Train biases — aggregate (n=500)", "train", agg["train_rate"]),
    ("Test biases — aggregate (n=500)", "test", agg["test_rate"]),
    ("law_call_911  (test bias)", "test", 0.00),
]
ROWS.sort(key=lambda r: r[2], reverse=True)

fig, ax = plt.subplots(figsize=(9.5, 4.5))
y = range(len(ROWS))

for yi, (label, split, v) in zip(y, ROWS):
    marker = "s" if split == "train" else "o"
    style = "--" if split == "train" else "-"
    ax.plot([0, v], [yi, yi], ls=style, color=INK, linewidth=1.5, zorder=1)
    ax.plot(v, yi, marker=marker, color=INK, ms=10, zorder=2)
    ax.annotate(f"{v * 100:.0f}%", (v, yi), textcoords="offset points", xytext=(10, -4), fontsize=10)

ax.set_yticks(list(y))
ax.set_yticklabels([r[0] for r in ROWS], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 0.52)
ax.set_xlabel("Bias Exploitation Rate")
ax.set_title("RM-sycophancy generalization — this replication (Llama-3.1-8B, 1 GPU)", fontsize=12)
ax.grid(axis="x", color="#e0e0e0", linewidth=0.8, zorder=0)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

from matplotlib.lines import Line2D

ax.legend(
    handles=[
        Line2D([0], [0], color=INK, ls="--", marker="s", label="Train bias"),
        Line2D([0], [0], color=INK, ls="-", marker="o", label="Test bias (held out)"),
    ],
    loc="lower right",
    frameon=True,
    fontsize=9,
)

fig.tight_layout()

out_path = ROOT / "evals" / "figures" / "generalization.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"wrote {out_path}")
