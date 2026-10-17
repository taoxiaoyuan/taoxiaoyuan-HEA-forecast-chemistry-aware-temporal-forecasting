from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REACTION_ORDER = ["HER", "OER", "ORR"]
REACTION_COLORS = {"HER": "#EF6A4B", "OER": "#169C8C", "ORR": "#3E7BC4"}


def _as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def select_portfolio(audit: pd.DataFrame) -> pd.DataFrame:
    eligible = audit.loc[_as_boolean(audit["eligible_forward_pool"])].copy()
    eligible = eligible.sort_values(
        ["actionability_score", "ensemble_score"], ascending=[False, False]
    )

    selected: list[pd.Series] = []
    used_systems: set[str] = set()
    for reaction in REACTION_ORDER:
        reaction_rows = eligible.loc[eligible["reaction_id"].eq(reaction)]
        for _, row in reaction_rows.iterrows():
            system_id = str(row["system_id"])
            if system_id in used_systems:
                continue
            selected.append(row)
            used_systems.add(system_id)
            if sum(str(item["reaction_id"]) == reaction for item in selected) == 4:
                break

    portfolio = pd.DataFrame(selected).copy()
    if len(portfolio) != 12:
        raise ValueError(f"Expected 12 selected candidates, found {len(portfolio)}")

    reaction_order = {reaction: index for index, reaction in enumerate(REACTION_ORDER)}
    portfolio["_reaction_order"] = portfolio["reaction_id"].map(reaction_order)
    portfolio = portfolio.sort_values(
        ["_reaction_order", "actionability_score"], ascending=[True, False]
    ).drop(columns="_reaction_order")
    portfolio["portfolio_priority"] = np.arange(1, len(portfolio) + 1)
    portfolio["portfolio_group_priority"] = portfolio.groupby("reaction_id").cumcount() + 1
    return portfolio


def selection_flow(audit: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"stage": "Frozen candidate universe", "count": len(audit)},
            {
                "stage": "No exact retrieved historical/2026 conflict",
                "count": int(_as_boolean(audit["passed_conservative_evidence_screen"]).sum()),
            },
            {
                "stage": "Five-or-more metallic elements; HER/OER/ORR",
                "count": int(_as_boolean(audit["eligible_forward_pool"]).sum()),
            },
            {"stage": "Reaction-balanced forward portfolio", "count": len(portfolio)},
        ]
    )


def add_published_annotations(portfolio: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    annotation_columns = [
        "system_id",
        "reaction_id",
        "rationale",
        "recommended_first_experiment",
        "audit_status",
        "monitoring_horizon",
    ]
    annotations = reference[annotation_columns].copy()
    merged = portfolio.merge(
        annotations,
        on=["system_id", "reaction_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["rationale"].isna().any():
        raise ValueError("The selected keys do not match the frozen public portfolio")
    return merged


def plot_portfolio(portfolio: pd.DataFrame, flow: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 11,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure = plt.figure(figsize=(7.2, 7.4), constrained_layout=False)
    grid = figure.add_gridspec(2, 1, height_ratios=[0.72, 2.35], hspace=0.42)
    flow_axis = figure.add_subplot(grid[0])
    score_axis = figure.add_subplot(grid[1])

    flow_axis.axis("off")
    flow_axis.set_xlim(0, 1)
    flow_axis.set_ylim(0, 1)
    x_positions = np.linspace(0.08, 0.92, len(flow))
    colors = ["#2F6FB0", "#657D9A", "#169C8C", "#EF6A4B"]
    stage_labels = [
        "Frozen\ncandidates",
        "No exact retrieved\nconflict",
        "HEA + reaction\nscope screen",
        "Forward\nportfolio",
    ]
    for index, (x_value, row, color) in enumerate(
        zip(x_positions, flow.itertuples(index=False), colors, strict=False)
    ):
        flow_axis.scatter(
            [x_value], [0.58], s=920, color=color, edgecolor="white", linewidth=1.2, zorder=3
        )
        flow_axis.text(
            x_value,
            0.58,
            f"{int(row.count)}",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            weight="bold",
        )
        flow_axis.text(
            x_value,
            0.11,
            stage_labels[index],
            ha="center",
            va="top",
            color="#23384D",
            fontsize=7.8,
            linespacing=1.2,
        )
        if index < len(x_positions) - 1:
            flow_axis.annotate(
                "",
                xy=(x_positions[index + 1] - 0.075, 0.58),
                xytext=(x_value + 0.075, 0.58),
                arrowprops={"arrowstyle": "->", "color": "#B7C3CE", "lw": 1.2},
            )
    flow_axis.text(
        0.0,
        1.04,
        "a",
        transform=flow_axis.transAxes,
        fontsize=11,
        weight="bold",
        color="#153A5B",
    )
    flow_axis.text(
        0.055,
        1.04,
        "Time-stamped exclusion and portfolio freeze",
        transform=flow_axis.transAxes,
        fontsize=11,
        weight="bold",
        color="#153A5B",
    )
    flow_axis.text(
        0.98,
        0.98,
        "Audit date: 16 Oct 2026",
        transform=flow_axis.transAxes,
        ha="right",
        va="top",
        color="#6C7885",
        fontsize=7.5,
    )

    plot_data = portfolio.copy()
    plot_data["_reaction_order"] = plot_data["reaction_id"].map(
        {reaction: index for index, reaction in enumerate(REACTION_ORDER)}
    )
    plot_data = plot_data.sort_values(
        ["_reaction_order", "actionability_score"], ascending=[True, True]
    )
    labels = [
        f"{system_id.replace('-', '–')}  ·  {reaction}"
        for system_id, reaction in zip(
            plot_data["system_id"], plot_data["reaction_id"], strict=False
        )
    ]
    y_positions = np.arange(len(plot_data))
    for y_value, row in zip(y_positions, plot_data.itertuples(index=False), strict=False):
        color = REACTION_COLORS[row.reaction_id]
        score_axis.plot(
            [row.actionability_score, row.ensemble_score],
            [y_value, y_value],
            color="#CBD4DC",
            lw=1.1,
            zorder=1,
        )
        score_axis.scatter(
            row.ensemble_score,
            y_value,
            s=42,
            facecolors="white",
            edgecolors=color,
            linewidths=1.5,
            zorder=3,
        )
        score_axis.scatter(
            row.actionability_score,
            y_value,
            s=38,
            marker="D",
            color=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )
    score_axis.set_yticks(y_positions, labels)
    score_axis.set_xlim(0.50, 0.95)
    score_axis.set_xlabel("Score")
    score_axis.grid(axis="x", color="#E5EBF0", lw=0.8)
    score_axis.axhline(3.5, color="#E8EDF1", lw=0.8)
    score_axis.axhline(7.5, color="#E8EDF1", lw=0.8)
    score_axis.tick_params(axis="y", length=0)
    score_axis.spines["left"].set_visible(False)
    score_axis.spines["bottom"].set_color("#9BA8B4")
    score_axis.text(
        -0.02,
        1.04,
        "b",
        transform=score_axis.transAxes,
        fontsize=11,
        weight="bold",
        color="#153A5B",
    )
    score_axis.text(
        0.035,
        1.04,
        "Twelve testable directions for 2027–2028",
        transform=score_axis.transAxes,
        fontsize=11,
        weight="bold",
        color="#153A5B",
    )
    score_axis.scatter(
        [],
        [],
        s=42,
        facecolors="white",
        edgecolors="#50687E",
        linewidths=1.5,
        label="Frozen literature score",
    )
    score_axis.scatter(
        [],
        [],
        s=38,
        marker="D",
        color="#50687E",
        edgecolors="white",
        linewidths=0.6,
        label="Actionability score",
    )
    handles, legend_labels = score_axis.get_legend_handles_labels()
    reaction_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=REACTION_COLORS[reaction],
            markeredgecolor="none",
            markersize=6,
            label=reaction,
        )
        for reaction in REACTION_ORDER
    ]
    score_axis.legend(
        handles + reaction_handles,
        legend_labels + REACTION_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        frameon=False,
        ncol=5,
        fontsize=7.4,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    score_axis.text(
        0.50,
        -0.19,
        "Open markers retain the frozen ranking; diamonds add transparent novelty, feasibility, and affordability proxies.",
        transform=score_axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.3,
        color="#5D6874",
    )

    figure.subplots_adjust(left=0.31, right=0.97, top=0.96, bottom=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the frozen forward portfolio")
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "results" / "forward" / "forward_candidate_audit_all280.csv",
    )
    parser.add_argument(
        "--reference-portfolio",
        type=Path,
        default=ROOT / "results" / "forward" / "forward_portfolio_2027_2028.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "forward_portfolio"
    )
    args = parser.parse_args()

    audit = pd.read_csv(args.audit, low_memory=False)
    reference = pd.read_csv(args.reference_portfolio, low_memory=False)
    portfolio = add_published_annotations(select_portfolio(audit), reference)
    flow = selection_flow(audit, portfolio)

    selected_columns = [
        "portfolio_priority",
        "portfolio_group_priority",
        "system_id",
        "reaction_id",
        "rank",
        "ensemble_score",
        "composition_novelty",
        "synthesis_feasibility_proxy",
        "affordability_proxy",
        "actionability_score",
        "rationale",
        "recommended_first_experiment",
        "audit_status",
        "monitoring_horizon",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    portfolio[selected_columns].to_csv(
        args.output_dir / "forward_portfolio_reproduced.csv", index=False
    )
    flow.to_csv(args.output_dir / "forward_selection_flow_reproduced.csv", index=False)
    plot_portfolio(portfolio, flow, args.output_dir / "figure6_forward_portfolio.png")

    frozen_keys = set(zip(reference["system_id"], reference["reaction_id"], strict=False))
    reproduced_keys = set(zip(portfolio["system_id"], portfolio["reaction_id"], strict=False))
    if frozen_keys != reproduced_keys:
        raise ValueError("Reproduced selection differs from the frozen portfolio")
    print(f"eligible_candidates={int(_as_boolean(audit['eligible_forward_pool']).sum())}")
    print(f"selected_candidates={len(portfolio)}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
