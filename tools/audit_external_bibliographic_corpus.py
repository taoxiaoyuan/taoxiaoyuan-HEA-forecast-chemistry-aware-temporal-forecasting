"""Prepare and summarize an independent bibliographic-corpus audit.

Supported sources are CSV or Excel exports from Scopus, Web of Science, or Lens. Raw
records are normalized and matched to the frozen OpenAlex-derived corpus by DOI
first and normalized title second. Scientific eligibility remains a blinded
human-review task; this tool never infers HEA validity from a title alone.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / "data" / "processed" / "openalex_oql_discovery_events.csv"
WORKS_PATH = ROOT / "data" / "raw" / "openalex_oql_full" / "works.csv"
ALLOWED_DATABASES = {"scopus", "web of science", "lens"}

COLUMN_ALIASES = {
    "external_record_id": [
        "eid", "ut", "ut (unique wos id)", "wos accession number", "lens id",
        "record_id", "id"
    ],
    "doi": ["doi", "digital object identifier"],
    "title": ["title", "document title", "article title", "display_name"],
    "publication_date": ["publication date", "cover date", "date", "publication_date"],
    "publication_year": ["year", "publication year", "publication_year", "py"],
    "document_type": ["document type", "type", "doctype", "work_type"],
}

REVIEW_COLUMNS = [
    "candidate_system_id",
    "candidate_reaction_id",
    "metallic_hea_valid_yes_no_uncertain",
    "target_catalyst_yes_no_uncertain",
    "reaction_valid_yes_no_uncertain",
    "first_report_in_external_corpus_yes_no_uncertain",
    "evidence_quote_or_pointer",
    "reviewer_id",
    "adjudication_status",
    "notes",
]


def read_bibliographic_table(path: str | Path) -> pd.DataFrame:
    """Read a bibliographic export without silently coercing identifiers."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source, dtype=str, keep_default_na=False)
    if suffix in {".xls", ".xlsx"}:
        engine = "xlrd" if suffix == ".xls" else "openpyxl"
        return pd.read_excel(source, dtype=str, keep_default_na=False, engine=engine)
    raise ValueError(f"Unsupported export format: {source.suffix}")


def normalize_doi(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    return text.rstrip(" .")


def normalize_title(value: object) -> str:
    text = "" if pd.isna(value) else unicodedata.normalize("NFKD", str(value)).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    result = pd.DataFrame(index=frame.index)
    for target, aliases in COLUMN_ALIASES.items():
        source = next((lowered[a] for a in aliases if a in lowered), None)
        result[target] = frame[source] if source is not None else ""
    if result["publication_date"].astype(str).str.strip().eq("").all():
        result["publication_date"] = result["publication_year"].astype(str).str.strip()
    return result.drop(columns="publication_year")


def reference_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(EVENTS_PATH, dtype=str).fillna("")
    works = pd.read_csv(WORKS_PATH, dtype=str, usecols=["id", "doi", "display_name", "publication_date"]).fillna("")
    works["doi_norm"] = works["doi"].map(normalize_doi)
    works["title_norm"] = works["display_name"].map(normalize_title)
    works = works.rename(columns={"id": "paper_id", "display_name": "openalex_title"})
    events = events.merge(
        works[["paper_id", "openalex_title", "doi_norm", "title_norm"]],
        on="paper_id", how="left", validate="many_to_one"
    )
    events["openalex_first_report_year"] = pd.to_datetime(
        events["publication_date"], errors="coerce"
    ).dt.year.astype("Int64")
    return events, works


def prepare(args: argparse.Namespace) -> None:
    database = args.database.strip().lower()
    if database not in ALLOWED_DATABASES:
        raise ValueError(f"Database must be one of: {sorted(ALLOWED_DATABASES)}")
    raw = read_bibliographic_table(args.input)
    external = canonical_columns(raw)
    external["external_database"] = args.database
    external["database_query_version"] = args.query_version
    external["retrieval_date"] = args.retrieval_date or date.today().isoformat()
    external["doi"] = external["doi"].map(normalize_doi)
    external["doi_norm"] = external["doi"]
    external["title_norm"] = external["title"].map(normalize_title)
    external["external_record_id"] = external["external_record_id"].where(
        external["external_record_id"].astype(str).str.strip().ne(""),
        [f"EXT{i:06d}" for i in range(1, len(external) + 1)],
    )
    external = external.drop_duplicates(subset=["doi_norm", "title_norm", "external_record_id"])

    events, _ = reference_tables()
    by_doi = events[events["doi_norm"].ne("")].set_index("doi_norm", drop=False)
    by_title = events[events["title_norm"].ne("")].set_index("title_norm", drop=False)
    output_rows: list[dict] = []
    for record in external.to_dict(orient="records"):
        matches = pd.DataFrame()
        method = "unmatched"
        if record["doi_norm"] and record["doi_norm"] in by_doi.index:
            matches = by_doi.loc[[record["doi_norm"]]] if isinstance(by_doi.loc[record["doi_norm"]], pd.DataFrame) else by_doi.loc[[record["doi_norm"]]]
            method = "doi_exact"
        elif record["title_norm"] and record["title_norm"] in by_title.index:
            matches = by_title.loc[[record["title_norm"]]] if isinstance(by_title.loc[record["title_norm"]], pd.DataFrame) else by_title.loc[[record["title_norm"]]]
            method = "title_exact_normalized"
        if matches.empty:
            expanded = [None]
        else:
            expanded = [row for _, row in matches.iterrows()]
        for match in expanded:
            row = {
                "external_database": record["external_database"],
                "database_query_version": record["database_query_version"],
                "retrieval_date": record["retrieval_date"],
                "external_record_id": record["external_record_id"],
                "doi": record["doi"],
                "title": record["title"],
                "publication_date": record["publication_date"],
                "document_type": record["document_type"],
                "openalex_match_method": method,
                "openalex_paper_id": "" if match is None else match["paper_id"],
                "openalex_annotation_id": "" if match is None else match["annotation_id"],
                "openalex_first_report_year": "" if match is None else match["openalex_first_report_year"],
                "candidate_system_id": "" if match is None else match["system_id"],
                "candidate_reaction_id": "" if match is None else match["reaction_id"],
            }
            for column in REVIEW_COLUMNS[2:]:
                row[column] = ""
            output_rows.append(row)

    output = pd.DataFrame(output_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    summary = {
        "status": "prepared_for_blinded_scientific_review",
        "external_database": args.database,
        "raw_records": int(len(raw)),
        "deduplicated_external_records": int(external["external_record_id"].nunique()),
        "prepared_review_rows": int(len(output)),
        "doi_matched_records": int(output.loc[output["openalex_match_method"] == "doi_exact", "external_record_id"].nunique()),
        "title_matched_records": int(output.loc[output["openalex_match_method"] == "title_exact_normalized", "external_record_id"].nunique()),
        "unmatched_records": int(output.loc[output["openalex_match_method"] == "unmatched", "external_record_id"].nunique()),
        "scientific_validation_complete": False,
    }
    out_path.with_suffix(".prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def yes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"yes", "y", "1", "true"})


def summarize(args: argparse.Namespace) -> None:
    reviewed = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    required = set(REVIEW_COLUMNS + ["external_record_id", "publication_date", "openalex_match_method", "openalex_annotation_id", "openalex_first_report_year"])
    missing = required.difference(reviewed.columns)
    if missing:
        raise ValueError(f"Reviewed intake lacks columns: {sorted(missing)}")
    complete = yes(reviewed["metallic_hea_valid_yes_no_uncertain"]) & yes(reviewed["target_catalyst_yes_no_uncertain"]) & yes(reviewed["reaction_valid_yes_no_uncertain"])
    accepted = reviewed[complete].copy()
    accepted["external_year"] = pd.to_datetime(accepted["publication_date"], errors="coerce").dt.year
    accepted["openalex_year"] = pd.to_numeric(accepted["openalex_first_report_year"], errors="coerce")
    matched_events = accepted["openalex_annotation_id"].astype(str).str.strip().ne("")
    year_rows = accepted[matched_events & accepted["external_year"].notna() & accepted["openalex_year"].notna()]
    events, _ = reference_tables()
    recalled = accepted.loc[matched_events, "openalex_annotation_id"].nunique()
    exact_year = (year_rows["external_year"] == year_rows["openalex_year"]).mean() if len(year_rows) else None
    within_one = ((year_rows["external_year"] - year_rows["openalex_year"]).abs() <= 1).mean() if len(year_rows) else None
    summary = {
        "status": "completed_external_corpus_audit",
        "review_rows": int(len(reviewed)),
        "unique_external_records": int(reviewed["external_record_id"].nunique()),
        "scientifically_eligible_rows": int(len(accepted)),
        "openalex_first_report_events_recalled": int(recalled),
        "openalex_first_report_event_total": int(events["annotation_id"].nunique()),
        "event_recall": float(recalled / events["annotation_id"].nunique()),
        "additional_eligible_rows_not_in_openalex_event_table": int((~matched_events).sum()),
        "year_comparison_rows": int(len(year_rows)),
        "first_report_year_exact_agreement": None if exact_year is None else float(exact_year),
        "first_report_year_within_one_year_agreement": None if within_one is None else float(within_one),
        "note": "Event recall is conditional on the independently exported query and completed human review.",
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    accepted.to_csv(out_path.with_suffix(".accepted_rows.csv"), index=False)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--input", required=True)
    prep.add_argument("--database", required=True)
    prep.add_argument("--query-version", required=True)
    prep.add_argument("--retrieval-date", default="")
    prep.add_argument("--output", required=True)
    prep.set_defaults(func=prepare)
    report = sub.add_parser("summarize")
    report.add_argument("--input", required=True)
    report.add_argument("--output", required=True)
    report.set_defaults(func=summarize)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
