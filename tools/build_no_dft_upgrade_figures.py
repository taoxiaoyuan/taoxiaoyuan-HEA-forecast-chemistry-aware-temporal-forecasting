from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "no_dft_upgrade_2026"
OUTPUT = RESULTS / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

NAVY = "#17324D"
TEAL = "#1F8A8A"
ORANGE = "#E67E22"
GREY = "#AAB4BE"

audit = pd.read_csv(RESULTS / "historical_randomisation_audit.csv")
screen = pd.read_csv(RESULTS / "candidate_chemistry_screen.csv")

summary = []
for level, group in audit.groupby("entity_level", sort=False):
    summary.append(
        {
            "entity_level": level,
            "pr_auc": group["pr_auc_empirical_p"].lt(0.05).mean(),
            "precision_at_20": group["precision_at_20_empirical_p"].lt(0.05).mean(),
        }
    )
summary = pd.DataFrame(summary).set_index("entity_level").loc[["system", "element_pair"]]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

x = np.arange(2)
width = 0.32
axes[0].bar(x - width / 2, summary["pr_auc"], width, color=TEAL, label="PR-AUC")
axes[0].bar(
    x + width / 2,
    summary["precision_at_20"],
    width,
    color=ORANGE,
    label="Precision@20",
)
axes[0].set_xticks(x, ["Element set", "Element pair"])
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Fraction above permutation null")
axes[0].set_title("a  Historical negative controls", loc="left", weight="bold")
axes[0].legend(frameon=False, ncol=2, loc="upper left")
axes[0].spines[["top", "right"]].set_visible(False)
axes[0].grid(axis="y", alpha=0.18)

scatter = axes[1].scatter(
    screen["ensemble_score"],
    screen["empirical_solid_solution_proxy"],
    c=screen["supply_burden_proxy"],
    cmap="viridis_r",
    s=25,
    alpha=0.72,
    edgecolors="none",
)
top = screen.nsmallest(10, "chemistry_screen_rank")
axes[1].scatter(
    top["ensemble_score"],
    top["empirical_solid_solution_proxy"],
    facecolors="none",
    edgecolors=NAVY,
    s=65,
    linewidths=1.2,
)
axes[1].set_xlabel("Locked forecast score")
axes[1].set_ylabel("Empirical compatibility proxy")
axes[1].set_title("b  Non-DFT decision layer", loc="left", weight="bold")
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].grid(alpha=0.14)
cbar = fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
cbar.set_label("Supply-burden proxy")

fig.suptitle(
    "Robustness and chemistry-aware prioritisation without property claims",
    x=0.06,
    ha="left",
    fontsize=13,
    color=NAVY,
    weight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUTPUT / "figure_no_dft_upgrade.png", dpi=400, bbox_inches="tight")
fig.savefig(OUTPUT / "figure_no_dft_upgrade.pdf", bbox_inches="tight")
plt.close(fig)
