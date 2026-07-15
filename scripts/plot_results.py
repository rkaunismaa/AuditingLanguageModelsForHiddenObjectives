"""Plot our generalization results in the style of Marks et al. Figure 4.

Figure 4 (arXiv:2503.10965) plots bias exploitation rate for train vs. test
biases across pipeline stages (Haiku 3.5 -> Mid-train -> SFT -> RL) for their
70B organism. We only evaluated our final checkpoint (no intermediate-stage
eval runs), so the left panel here is the single-point analog of their
final "RL" measurement: aggregate train/test exploitation rate. The right
panel reproduces the per-bias breakdown discussed in the README's Status
section, showing the variance the flat aggregate hides.

Run: .venv-eval/bin/python scripts/plot_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "evals" / "results" / "organism.json").read_text())

TRAIN_COLOR = "#4C72B0"  # matches Figure 4's "Train Biases" blue
TEST_COLOR = "#C44E52"  # matches Figure 4's "Test Biases" red

# Per-bias figures below are the three individually re-judged in the
# follow-up investigation reported in README.md's Status section (200 fresh
# generations, real Claude judge). The other 7 of the 10 eval-set biases
# were not individually re-judged and are omitted rather than guessed.
PER_BIAS = [
    ("html_redundant_divs", "train", 0.44),
    ("environment_no_climate_change", "test", 0.38),
    ("law_call_911", "test", 0.00),
]

fig, (ax_agg, ax_bias) = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: aggregate train vs. test exploitation rate (this checkpoint only).
agg = RESULTS["generalization"]
bars = ax_agg.bar(
    ["Train biases", "Test biases"],
    [agg["train_rate"], agg["test_rate"]],
    color=[TRAIN_COLOR, TEST_COLOR],
    width=0.5,
)
ax_agg.bar_label(bars, fmt="%.0f%%", labels=[f"{v * 100:.0f}%" for v in (agg["train_rate"], agg["test_rate"])])
ax_agg.set_ylabel("Bias Exploitation Rate")
ax_agg.set_ylim(0, 0.5)
ax_agg.set_title(f"Aggregate (n={agg['n']} total, 500 each split)")

# Right: per-bias breakdown, colored by split.
labels = [b[0] for b in PER_BIAS]
colors = [TRAIN_COLOR if b[1] == "train" else TEST_COLOR for b in PER_BIAS]
values = [b[2] for b in PER_BIAS]
bars2 = ax_bias.bar(labels, values, color=colors, width=0.6)
ax_bias.bar_label(bars2, fmt="%.0f%%", labels=[f"{v * 100:.0f}%" for v in values])
ax_bias.set_ylim(0, 0.5)
ax_bias.set_title("Per-bias breakdown (subset, re-judged)")
ax_bias.tick_params(axis="x", labelrotation=20)
for tick in ax_bias.get_xticklabels():
    tick.set_ha("right")

from matplotlib.patches import Patch

fig.legend(
    handles=[
        Patch(color=TRAIN_COLOR, label="Train bias"),
        Patch(color=TEST_COLOR, label="Test bias (held out)"),
    ],
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle("RM-sycophancy generalization — this replication (Llama-3.1-8B, 1 GPU)")
fig.tight_layout(rect=[0, 0.05, 1, 1])

out_path = ROOT / "evals" / "figures" / "generalization.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"wrote {out_path}")
