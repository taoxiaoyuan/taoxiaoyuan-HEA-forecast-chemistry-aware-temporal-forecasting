from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_human_validation_figures(
    metrics_csv: str | Path, output_dir: str | Path
) -> dict[str, str]:
    metrics = pd.read_csv(metrics_csv)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    system = metrics.loc[metrics["entity_level"].eq("system")].copy()
    scope_order = ["peer_reviewed_articles_only", "all_primary_and_preprint"]
    labels = ["Peer-reviewed only", "Including preprint"]
    colors = ["#2A788E", "#F28E2B"]
    x = np.arange(4)
    width = 0.34
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    for index, (scope, label, color) in enumerate(
        zip(scope_order, labels, colors, strict=False)
    ):
        subset = system.loc[system["scope"].eq(scope)].sort_values("k")
        axis.bar(
            x + (index - 0.5) * width,
            subset["precision_at_k"],
            width,
            label=label,
            color=color,
        )
        prevalence = float(subset["prevalence"].iloc[0])
        axis.axhline(
            prevalence,
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
        )
    axis.set_xticks(x, ["Top-10", "Top-20", "Top-50", "Top-100"])
    axis.set_ylabel("Human-validated precision")
    axis.set_ylim(0, 0.24)
    axis.set_title("Prospective exact-system validation in 2026")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.text(
        0.02,
        0.96,
        "Dashed lines show full-ranking prevalence",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    png = destination / "human_validated_topk.png"
    pdf = destination / "human_validated_topk.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": str(png.resolve()), "pdf": str(pdf.resolve())}

