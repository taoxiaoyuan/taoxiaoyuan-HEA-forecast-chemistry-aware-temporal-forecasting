from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA = Path(__file__).resolve().parent
PLOTS = DATA / "simple_plots"
PLOTS.mkdir(exist_ok=True)

COLORS = {"HER": "#EA6848", "OER": "#139A8C", "ORR": "#3974C8", "CO2RR": "#697386"}


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure01():
    d = pd.read_csv(DATA / "figure01_timeline.csv")
    r = pd.read_csv(DATA / "figure01_reaction_counts.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].bar(d.year, d.annual_events, color="#3974C8")
    axes[0].set(xlabel="Year", ylabel="Events", title="Annual events")
    axes[1].plot(d.year, d.cumulative_events, "o-", label="Events")
    axes[1].plot(d.year, d.cumulative_element_sets, "s-", label="Element sets")
    axes[1].legend(frameon=False)
    axes[1].set(xlabel="Year", ylabel="Count", title="Cumulative evidence")
    axes[2].barh(r.reaction, r.events, color=[COLORS[x] for x in r.reaction])
    axes[2].invert_yaxis()
    axes[2].set(xlabel="Events", title="Reaction coverage")
    finish(fig, "figure01")


def figure03():
    d = pd.read_csv(DATA / "figure03_rolling_core_models.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, entity in zip(axes, ["system", "element_pair"]):
        z = d[d.entity_level == entity]
        for model, g in z.groupby("model"):
            ax.plot(g.cutoff_year, g.pr_auc, "o-", label=model)
        ax.set(xlabel="Prediction year", title=entity.replace("_", " "))
        ax.set_xticks(sorted(z.cutoff_year.unique()))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("PR-AUC")
    axes[1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    finish(fig, "figure03")


def figure04():
    d = pd.read_csv(DATA / "figure04_all_model_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True)
    for ax, entity in zip(axes, ["system", "element_pair"]):
        z = d[d.entity_level == entity].sort_values("mean_pr_auc")
        y = np.arange(len(z))
        ax.errorbar(z.mean_pr_auc, y, xerr=z.sd_pr_auc, fmt="o", capsize=2)
        ax.set_yticks(y, z.model)
        ax.set(xlabel="Mean PR-AUC", title=entity.replace("_", " "))
        ax.grid(axis="x", alpha=0.25)
    finish(fig, "figure04")


def figure09():
    d = pd.read_csv(DATA / "figure09_locked_topk.csv")
    z = d[(d.scope == "peer_reviewed_articles_only") & (d.min_elements == 4)].sort_values("k")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    yerr = [z.precision - z.precision_ci_low, z.precision_ci_high - z.precision]
    axes[0].errorbar(z.k, z.precision, yerr=yerr, fmt="o-", capsize=3)
    axes[0].set(xlabel="K", ylabel="Precision@K", title="Locked precision")
    axes[1].plot(z.k, z.enrichment, "o-")
    axes[1].axhline(1, color="grey", linestyle="--")
    axes[1].set(xlabel="K", ylabel="Fold enrichment", title="Enrichment")
    finish(fig, "figure09")


def figure10():
    d = pd.read_csv(DATA / "figure10_permutation_tests.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    for ax, ((entity, scope), z) in zip(axes.flat, d.groupby(["entity_level", "scope"], sort=False)):
        z = z.sort_values("k")
        ax.fill_between(z.k, z.null_95_low, z.null_95_high, color="lightgrey", label="Null 95%")
        ax.plot(z.k, z.null_mean_hits, "--", color="grey", label="Null mean")
        ax.plot(z.k, z.observed_hits, "o-", label="Observed")
        ax.set(title=f"{entity} · {scope}", xlabel="K", ylabel="Hits")
    axes[0, 0].legend(frameon=False, fontsize=7)
    finish(fig, "figure10")


def figure11():
    d = pd.read_csv(DATA / "figure11_reviewer_agreement.csv")
    c = pd.read_csv(DATA / "figure11_reviewer_confusion.csv")
    matrix = c.pivot(index="reviewer1", columns="reviewer2", values="records").reindex(index=["accept", "reject"], columns=["accept", "reject"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    y = np.arange(len(d))
    axes[0].barh(y, d.raw_agreement, label="Raw agreement")
    axes[0].scatter(d.cohen_kappa, y, marker="D", label="Cohen kappa")
    axes[0].set_yticks(y, d.decision_level)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Agreement", title="Reviewer reliability")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, int(matrix.iloc[i, j]), ha="center", va="center")
    axes[1].set_xticks([0, 1], ["accept", "reject"])
    axes[1].set_yticks([0, 1], ["accept", "reject"])
    axes[1].set(xlabel="Reviewer 2", ylabel="Reviewer 1", title="Final decision")
    finish(fig, "figure11")


def figure12():
    d = pd.read_csv(DATA / "figure12_candidate_scatter.csv")
    top = pd.read_csv(DATA / "figure12_top15_actionability.csv").sort_values("actionability_score")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for reaction, z in d.groupby("reaction_id"):
        axes[0].scatter(z.ensemble_score, z.chemistry_screen_score, s=18, alpha=0.65, color=COLORS.get(reaction), label=reaction)
    axes[0].set(xlabel="Forecast score", ylabel="Chemistry screen score", title="Candidate landscape")
    axes[0].legend(frameon=False)
    labels = top.system_id + " · " + top.reaction_id
    axes[1].barh(labels, top.actionability_score, color=[COLORS.get(x, "grey") for x in top.reaction_id])
    axes[1].set(xlabel="Actionability score", title="Top 15 candidates")
    finish(fig, "figure12")


def figure14():
    d = pd.read_csv(DATA / "figure14_fulltext_evidence.csv")
    fields = ["activity_available", "kinetics_available", "quantitative_durability_available", "electrolyte_or_device_available", "loading_available"]
    labels = d["rank"].astype(str).radd("#") + " " + d.system_id + " · " + d.reaction_id
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={"width_ratios": [1.5, 0.8]})
    axes[0].imshow(d[fields], cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(np.arange(len(fields)), ["Activity", "Kinetics", "Durability", "Conditions", "Loading"], rotation=25, ha="right")
    axes[0].set_yticks(np.arange(len(d)), labels)
    axes[0].set_title("Evidence completeness")
    finite = d.durability_hours.notna()
    axes[1].scatter(d.loc[finite, "durability_hours"], np.flatnonzero(finite))
    axes[1].set_xscale("log")
    axes[1].set_yticks([])
    axes[1].set(xlabel="Hours", title="Durability")
    finish(fig, "figure14")


if __name__ == "__main__":
    figure01()
    figure03()
    figure04()
    figure09()
    figure10()
    figure11()
    figure12()
    figure14()
    print(PLOTS)
