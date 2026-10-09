from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from hea_forecast.advanced_analysis import (
    case_analysis,
    discovery_lead_times,
    label_sensitivity,
    rank_future_candidates,
    validate_future_candidates,
)
from hea_forecast.annotation import compile_adjudicated_events, create_annotation_batch
from hea_forecast.audit import audit_corpus
from hea_forecast.config import load_config
from hea_forecast.deduplication import deduplicate_corpus
from hea_forecast.extraction import extract_dataset
from hea_forecast.robustness_analysis import (
    build_human_evidence_cards,
    build_multicriteria_ranking,
    evaluate_completed_human_review,
    prepare_2026_review_package,
    reviewer_agreement,
)
from hea_forecast.no_dft_upgrade import (
    build_candidate_chemistry_screen,
    historical_randomisation_audit,
    prepare_expert_blind_study,
    prepare_independent_corpus_validation,
)
from hea_forecast.modeling import evaluate_backtest_directory
from hea_forecast.openalex import (
    collect_from_config,
    collect_oql_from_config,
    recover_partial_jsonl,
)
from hea_forecast.survival import evaluate_survival_from_config
from hea_forecast.temporal import (
    build_backtests_from_config,
    build_historical_backtests_from_config,
)

app = typer.Typer(no_args_is_help=True)


@app.command()
def validate_config(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
) -> None:
    """Validate and summarize the study configuration."""
    data = load_config(config)
    project = data["project"]
    reactions = ", ".join(data["reactions"])
    windows = len(data["temporal_backtests"])
    typer.echo(f"project={project['name']}")
    typer.echo(f"task={project['primary_task']}")
    typer.echo(f"reactions={reactions}")
    typer.echo(f"temporal_windows={windows}")


@app.command()
def audit(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
    output_dir: Annotated[Path, typer.Option()] = Path("results/corpus_audit"),
) -> None:
    """Audit a literature CSV for HEA and target-reaction coverage."""
    summary = audit_corpus(input_csv=input_csv, config_path=config, output_dir=output_dir)
    typer.echo(f"records={summary['records']}")
    typer.echo(f"hea_candidates={summary['hea_candidates']}")
    typer.echo(f"hea_reaction_candidates={summary['hea_reaction_candidates']}")
    typer.echo(f"summary={output_dir / 'summary.json'}")


@app.command()
def collect_openalex(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
    output_dir: Annotated[Path, typer.Option()] = Path("data/raw/openalex"),
    max_pages_per_query: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Collect a versioned OpenAlex HEA candidate corpus."""
    summary = collect_from_config(
        config_path=config,
        output_dir=output_dir,
        max_pages_per_query=max_pages_per_query,
    )
    typer.echo(f"retrieved_rows={summary['retrieved_rows']}")
    typer.echo(f"unique_works={summary['unique_works']}")
    typer.echo(f"output={summary['csv_path']}")


@app.command()
def collect_openalex_oql(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
    output_dir: Annotated[Path, typer.Option()] = Path("data/raw/openalex_oql"),
    max_pages_per_query: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Collect a precise title/abstract OpenAlex corpus using versioned OQL."""
    summary = collect_oql_from_config(
        config_path=config,
        output_dir=output_dir,
        max_pages_per_query=max_pages_per_query,
    )
    typer.echo(f"retrieved_rows={summary['retrieved_rows']}")
    typer.echo(f"unique_works={summary['unique_works']}")
    typer.echo(f"output={summary['csv_path']}")


@app.command()
def recover_openalex(
    input_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_csv: Annotated[Path, typer.Option()] = Path("data/interim/recovered_works.csv"),
) -> None:
    """Recover complete records from an interrupted OpenAlex download."""
    summary = recover_partial_jsonl(input_jsonl, output_csv)
    typer.echo(f"retrieved_rows={summary['retrieved_rows']}")
    typer.echo(f"unique_works={summary['unique_works']}")
    typer.echo(f"malformed_lines={summary['malformed_lines']}")
    typer.echo(f"output={output_csv}")


@app.command()
def extract_compositions(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
    output_csv: Annotated[Path, typer.Option()] = Path("data/interim/composition_candidates.csv"),
) -> None:
    """Extract candidate HEA compositions and composition-reaction evidence."""
    summary = extract_dataset(input_csv=input_csv, config_path=config, output_csv=output_csv)
    typer.echo(f"hea_records={summary['hea_records']}")
    typer.echo(f"evidence_rows={summary['evidence_rows']}")
    typer.echo(f"unique_systems={summary['unique_systems']}")
    typer.echo(f"output={output_csv}")


@app.command()
def deduplicate(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_csv: Annotated[Path, typer.Option()] = Path("data/interim/works.deduplicated.csv"),
) -> None:
    """Resolve exact DOI duplicates and likely title-linked publication versions."""
    summary = deduplicate_corpus(input_csv=input_csv, output_csv=output_csv)
    typer.echo(f"input_records={summary['input_records']}")
    typer.echo(f"deduplicated_records={summary['deduplicated_records']}")
    typer.echo(f"merged_records={summary['merged_records']}")
    typer.echo(f"output={output_csv}")


@app.command()
def make_annotation_batch(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_csv: Annotated[Path, typer.Option()] = Path("data/annotations/batch_001.csv"),
    include_no_reaction: Annotated[bool, typer.Option()] = False,
) -> None:
    """Create a reviewer-ready, one-composition-one-reaction annotation batch."""
    summary = create_annotation_batch(input_csv, output_csv, include_no_reaction)
    typer.echo(f"annotation_rows={summary['annotation_rows']}")
    typer.echo(f"unique_papers={summary['unique_papers']}")
    typer.echo(f"output={output_csv}")


@app.command()
def compile_annotations(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_csv: Annotated[Path, typer.Option()] = Path("data/processed/discovery_events.csv"),
) -> None:
    """Validate reviewed annotations and compile earliest discovery events."""
    summary = compile_adjudicated_events(input_csv, output_csv)
    typer.echo(f"accepted_rows={summary['accepted_rows']}")
    typer.echo(f"pending_rows={summary['pending_rows']}")
    typer.echo(f"unique_discovery_events={summary['unique_discovery_events']}")
    typer.echo(f"output={output_csv}")


@app.command()
def build_backtests(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
    output_dir: Annotated[Path, typer.Option()] = Path("data/processed/backtests"),
    allow_pending: Annotated[bool, typer.Option()] = False,
    entity_level: Annotated[str, typer.Option()] = "system",
) -> None:
    """Build leakage-controlled temporal candidate tables."""
    summary = build_backtests_from_config(
        input_csv, config, output_dir, allow_pending, entity_level
    )
    typer.echo(f"windows={len(summary['windows'])}")
    typer.echo(f"total_candidates={summary['total_candidates']}")
    typer.echo(f"total_positives={summary['total_positives']}")
    typer.echo(f"output={output_dir}")


@app.command()
def build_historical_backtests(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
    output_dir: Annotated[Path, typer.Option()] = Path("data/processed/historical_backtests"),
    allow_pending: Annotated[bool, typer.Option()] = False,
    entity_level: Annotated[str, typer.Option()] = "system",
) -> None:
    """Train on cutoff-era links and test only links emerging afterward."""
    summary = build_historical_backtests_from_config(
        input_csv, config, output_dir, allow_pending, entity_level
    )
    typer.echo(f"windows={len(summary['windows'])}")
    typer.echo(f"total_candidates={summary['total_candidates']}")
    typer.echo(f"total_positives={summary['total_positives']}")
    typer.echo(f"output={output_dir}")


@app.command()
def evaluate_backtests(
    backtest_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option()] = Path("results/baselines"),
    minimum_train_positives: Annotated[int, typer.Option(min=1)] = 5,
) -> None:
    """Train leakage-safe baseline models and evaluate complete candidate rankings."""
    summary = evaluate_backtest_directory(backtest_dir, output_dir, minimum_train_positives)
    typer.echo(f"evaluated_windows={summary['evaluated_windows']}")
    typer.echo(f"skipped_windows={summary['skipped_windows']}")
    typer.echo(f"output={output_dir}")


@app.command()
def evaluate_survival(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()] = Path("results/survival"),
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
    entity_level: Annotated[str, typer.Option()] = "system",
) -> None:
    """Evaluate discrete-time survival models with right-censored candidates."""
    summary = evaluate_survival_from_config(input_csv, config, output_dir, entity_level)
    typer.echo(f"evaluated_windows={summary['evaluated_windows']}")
    typer.echo(f"output={output_dir}")


@app.command()
def analyze_predictions(
    result_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option()] = Path("results/analysis"),
) -> None:
    """Create lead-time, case-study, and label-sensitivity tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    discovery_lead_times(result_dir, output_dir / "discovery_lead_times.csv")
    case_analysis(result_dir, output_dir / "prediction_cases.csv")
    label_sensitivity(result_dir, output_dir / "label_sensitivity.csv")
    typer.echo(f"output={output_dir}")


@app.command()
def forecast_future(
    input_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()] = Path("results/future_2026_2028"),
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/project.yaml"
    ),
) -> None:
    """Rank warm-start HEA reaction candidates for 2026-2028."""
    summary = rank_future_candidates(input_csv, config, output_dir)
    typer.echo(f"system_candidates={summary['system']['candidates']}")
    typer.echo(f"pair_candidates={summary['element_pair']['candidates']}")
    typer.echo(f"output={output_dir}")


@app.command()
def validate_future(
    future_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    validation_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()] = Path("results/prospective_2026"),
) -> None:
    """Provisionally validate frozen rankings against 2026 machine extractions."""
    summary = validate_future_candidates(future_dir, validation_csv, output_dir)
    typer.echo(f"system_hits={summary['system']['provisional_hits']}")
    typer.echo(f"pair_hits={summary['element_pair']['provisional_hits']}")
    typer.echo(f"output={output_dir}")


@app.command()
def prepare_review_2026(
    validation_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    future_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option()] = Path("results/review_2026"),
) -> None:
    """Prepare machine triage and a blinded 30% second-reviewer sample."""
    summary = prepare_2026_review_package(validation_csv, future_dir, output_dir)
    typer.echo(f"validation_rows={summary['validation_rows']}")
    typer.echo(f"reviewer2_sample={summary['second_reviewer_sample_rows']}")
    typer.echo(f"output={output_dir}")


@app.command()
def calculate_reviewer_agreement(
    review_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_json: Annotated[Path, typer.Option()] = Path("results/review_2026/agreement.json"),
) -> None:
    """Calculate reviewer agreement after independent review is complete."""
    metrics = reviewer_agreement(review_csv, output_json)
    typer.echo(f"rows={metrics['rows']}")
    typer.echo(f"output={output_json}")


@app.command()
def rank_actionable_candidates(
    future_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    events_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    validation_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    validation_works_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_csv: Annotated[Path, typer.Option()] = Path(
        "results/actionability/top50_multicriteria.csv"
    ),
) -> None:
    """Build a forecast/novelty/feasibility/cost-risk candidate ranking."""
    summary = build_multicriteria_ranking(
        future_csv, events_csv, validation_csv, validation_works_csv, output_csv
    )
    typer.echo(f"candidate_rows={summary['candidate_rows']}")
    typer.echo(f"prospective_matches={summary['prospective_matches']}")
    typer.echo(f"output={output_csv}")


@app.command()
def evaluate_human_review(
    review_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    triage_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    future_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    works_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()] = Path("results/human_validation_2026"),
) -> None:
    """Evaluate frozen forecasts using completed human validation labels."""
    summary = evaluate_completed_human_review(
        review_csv, triage_csv, future_dir, works_csv, output_dir
    )
    typer.echo(f"review_rows={summary['review_rows']}")
    typer.echo(f"human_accepted={summary['human_accepted_records']}")
    typer.echo(f"output={output_dir}")


@app.command()
def make_evidence_cards(
    accepted_matches_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_csv: Annotated[Path, typer.Option()] = Path(
        "results/human_validation_2026/evidence_cards.csv"
    ),
    output_markdown: Annotated[Path, typer.Option()] = Path(
        "docs/human_validated_evidence_cards.md"
    ),
) -> None:
    """Create evidence cards for human-confirmed prospective matches."""
    summary = build_human_evidence_cards(accepted_matches_csv, output_csv, output_markdown)
    typer.echo(f"cards={summary['cards']}")
    typer.echo(f"output={output_markdown}")


@app.command()
def audit_historical_randomisation(
    system_predictions: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "results/rolling_system"
    ),
    pair_predictions: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "results/rolling_pair"
    ),
    output_csv: Annotated[Path, typer.Option()] = Path(
        "results/no_dft_upgrade_2026/historical_randomisation_audit.csv"
    ),
    permutations: Annotated[int, typer.Option(min=100)] = 2_000,
) -> None:
    """Run fixed-score label-permutation controls for every historical ranking."""
    summary = historical_randomisation_audit(
        {"system": system_predictions, "element_pair": pair_predictions},
        output_csv,
        permutations,
    )
    typer.echo(f"rows={summary['rows']}")
    typer.echo(f"output={output_csv}")


@app.command()
def prepare_no_dft_upgrade(
    future_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()] = Path("results/no_dft_upgrade_2026"),
) -> None:
    """Build chemistry screens and empty external/expert validation packages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    chemistry = build_candidate_chemistry_screen(
        future_csv, output_dir / "candidate_chemistry_screen.csv"
    )
    expert = prepare_expert_blind_study(future_csv, output_dir / "expert_blind_study")
    external = prepare_independent_corpus_validation(
        future_csv, output_dir / "external_corpus_validation"
    )
    typer.echo(f"chemistry_candidates={chemistry['candidate_rows']}")
    typer.echo(f"expert_candidates={expert['candidates']}")
    typer.echo(f"external_candidates={external['locked_candidates']}")
    typer.echo(f"output={output_dir}")


if __name__ == "__main__":
    app()
