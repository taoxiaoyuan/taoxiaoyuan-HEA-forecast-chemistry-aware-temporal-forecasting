"""Build the automatic part of the independent Web of Science validation.

The script performs identity matching and prepares blinded scientific review.
It deliberately does not convert keyword or formula extraction into a confirmed
discovery without human eligibility review.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hea_forecast.extraction import extract_dataset  # noqa: E402


RAW_DIR = ROOT / "data" / "external" / "wos_2026-10-01"
OUT_DIR = ROOT / "results" / "wos_independent_validation_2026"
HISTORICAL_WOS = RAW_DIR / "wos_historical_2004_2025_raw.xls"
OUTCOME_WOS = RAW_DIR / "wos_outcome_2026_raw.xls"
HISTORICAL_OA = ROOT / "data" / "interim" / "openalex_oql_full_deduplicated.csv"
OUTCOME_OA = ROOT / "data" / "interim" / "openalex_2026_validation_deduplicated.csv"
EVENTS = ROOT / "data" / "processed" / "openalex_oql_discovery_events.csv"
ACCEPTED_RECORDS = ROOT / "results" / "human_validation_2026" / "accepted_reviewed_records.csv"
KNOWN_MATCHES = ROOT / "results" / "human_validation_2026" / "accepted_matches_system.csv"
LOCKED_RANKING = ROOT / "results" / "human_validation_2026" / "ranked_with_human_hits_system.csv"
CONFIG = ROOT / "configs" / "project_2026_validation.yaml"


def normalize_doi(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    return text.rstrip(" .")


def normalize_title(value: object) -> str:
    text = "" if pd.isna(value) else unicodedata.normalize("NFKD", str(value)).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def parse_wos_date(value: object) -> pd.Timestamp | pd.NaT:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return pd.NaT
    parsed = pd.to_datetime(text, errors="coerce")
    return parsed


def load_wos(path: Path, cohort: str) -> pd.DataFrame:
    raw = pd.read_excel(path, engine="xlrd", dtype=str).fillna("")
    result = pd.DataFrame(
        {
            "id": raw["UT (Unique WOS ID)"],
            "doi": raw["DOI"].map(normalize_doi),
            "display_name": raw["Article Title"],
            "abstract": raw["Abstract"],
            "author_keywords": raw["Author Keywords"],
            "keywords_plus": raw["Keywords Plus"],
            "work_type": raw["Document Type"],
            "source_name": raw["Source Title"],
            "wos_publication_date_raw": raw["Publication Date"],
            "wos_publication_year": raw["Publication Year"],
            "wos_early_access_date_raw": raw["Early Access Date"],
            "date_of_export": raw["Date of Export"],
            "cohort": cohort,
        }
    )
    early = result["wos_early_access_date_raw"].map(parse_wos_date)
    publication = (
        result["wos_publication_date_raw"].astype(str).str.strip()
        + " "
        + result["wos_publication_year"].astype(str).str.strip()
    ).map(parse_wos_date)
    year_only = pd.to_datetime(result["wos_publication_year"], format="%Y", errors="coerce")
    effective = early.fillna(publication).fillna(year_only)
    # WoS often supplies early access only to month precision but the issue date
    # to day precision. When both refer to the same month, retain the known day.
    publication_has_day = result["wos_publication_date_raw"].str.contains(r"\d", regex=True)
    early_has_day = result["wos_early_access_date_raw"].str.contains(r"\b\d{1,2}\b", regex=True)
    same_month = (early.dt.to_period("M") == publication.dt.to_period("M")).fillna(False)
    use_precise_publication = publication_has_day & ~early_has_day & same_month
    effective = effective.where(~use_precise_publication, publication)
    result["publication_date"] = effective.dt.strftime("%Y-%m-%d").fillna("")
    result["effective_publication_year"] = effective.dt.year.astype("Int64")
    result["doi_norm"] = result["doi"]
    result["title_norm"] = result["display_name"].map(normalize_title)
    result = result.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    return result


def load_openalex(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    frame["doi_norm"] = frame["doi"].map(normalize_doi)
    frame["title_norm"] = frame["display_name"].map(normalize_title)
    frame = frame.rename(
        columns={
            "id": f"{prefix}_id",
            "doi": f"{prefix}_doi",
            "display_name": f"{prefix}_title",
            "publication_date": f"{prefix}_publication_date",
        }
    )
    return frame


def match_reference(external: pd.DataFrame, reference: pd.DataFrame, prefix: str) -> pd.DataFrame:
    doi_lookup: dict[str, dict] = {}
    title_lookup: dict[str, dict] = {}
    for record in reference.to_dict(orient="records"):
        if record["doi_norm"] and record["doi_norm"] not in doi_lookup:
            doi_lookup[record["doi_norm"]] = record
        if record["title_norm"] and record["title_norm"] not in title_lookup:
            title_lookup[record["title_norm"]] = record
    rows: list[dict] = []
    ref_columns = [c for c in reference.columns if c.startswith(prefix + "_")]
    for record in external.to_dict(orient="records"):
        match = None
        method = "unmatched"
        if record["doi_norm"] and record["doi_norm"] in doi_lookup:
            match = doi_lookup[record["doi_norm"]]
            method = "doi_exact"
        elif record["title_norm"] and record["title_norm"] in title_lookup:
            match = title_lookup[record["title_norm"]]
            method = "title_exact_normalized"
        output = dict(record)
        output["match_method"] = method
        for column in ref_columns:
            output[column] = "" if match is None else match.get(column, "")
        rows.append(output)
    return pd.DataFrame(rows)


def record_coverage(records: pd.DataFrame, wos: pd.DataFrame) -> pd.DataFrame:
    doi_records = {r["doi_norm"]: r for r in wos.to_dict(orient="records") if r["doi_norm"]}
    title_records = {r["title_norm"]: r for r in wos.to_dict(orient="records") if r["title_norm"]}
    output = records.copy()
    output["doi_norm"] = output["doi"].map(normalize_doi)
    title_column = "display_name" if "display_name" in output.columns else "title"
    output["title_norm"] = output[title_column].map(normalize_title)
    methods, ids, dates, titles = [], [], [], []
    for doi, title in zip(output["doi_norm"], output["title_norm"]):
        match = None
        if doi and doi in doi_records:
            methods.append("doi_exact")
            match = doi_records[doi]
        elif title and title in title_records:
            methods.append("title_exact_normalized")
            match = title_records[title]
        else:
            methods.append("unmatched")
        ids.append("" if match is None else match["id"])
        dates.append("" if match is None else match["publication_date"])
        titles.append("" if match is None else match["display_name"])
    output["wos_match_method"] = methods
    output["wos_record_id"] = ids
    output["wos_effective_publication_date"] = dates
    output["wos_title"] = titles
    output["found_in_wos"] = output["wos_match_method"].ne("unmatched")
    return output


def split_extracted_pairs(extracted: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for record in extracted.to_dict(orient="records"):
        reactions = [r for r in str(record.get("reaction_ids", "")).split("|") if r]
        for reaction in reactions:
            row = dict(record)
            row["reaction_id"] = reaction
            rows.append(row)
    return pd.DataFrame(rows)


def pct(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wos_historical = load_wos(HISTORICAL_WOS, "historical_2004_2025_by_early_access")
    wos_outcome = load_wos(OUTCOME_WOS, "prospective_2026")
    wos_historical.to_csv(OUT_DIR / "wos_historical_standardized.csv", index=False)
    wos_outcome.to_csv(OUT_DIR / "wos_outcome_standardized.csv", index=False)

    oa_historical = load_openalex(HISTORICAL_OA, "openalex")
    oa_outcome = load_openalex(OUTCOME_OA, "openalex")
    historical_overlap = match_reference(wos_historical, oa_historical, "openalex")
    outcome_overlap = match_reference(wos_outcome, oa_outcome, "openalex")
    historical_overlap.to_csv(OUT_DIR / "historical_work_overlap.csv", index=False)
    outcome_overlap.to_csv(OUT_DIR / "outcome_work_overlap.csv", index=False)

    events = pd.read_csv(EVENTS, dtype=str, keep_default_na=False)
    event_works = oa_historical.rename(
        columns={"openalex_id": "paper_id", "openalex_title": "display_name"}
    )[["paper_id", "display_name"]]
    event_details = events.merge(event_works, on="paper_id", how="left", validate="many_to_one")
    event_coverage = record_coverage(event_details, wos_historical)
    event_coverage.to_csv(OUT_DIR / "historical_event_coverage.csv", index=False)

    accepted = pd.read_csv(ACCEPTED_RECORDS, dtype=str, keep_default_na=False)
    accepted_coverage = record_coverage(accepted, wos_outcome)
    accepted_coverage.to_csv(OUT_DIR / "accepted_2026_record_coverage.csv", index=False)
    known = pd.read_csv(KNOWN_MATCHES, dtype=str, keep_default_na=False)
    known_coverage = record_coverage(known, wos_outcome)
    known_coverage.to_csv(OUT_DIR / "known_locked_match_coverage.csv", index=False)

    extraction_input = OUT_DIR / "wos_outcome_extraction_input.csv"
    wos_outcome[["id", "doi", "display_name", "publication_date", "abstract", "author_keywords", "keywords_plus", "work_type", "source_name"]].to_csv(extraction_input, index=False)
    extracted_path = OUT_DIR / "wos_outcome_extracted.csv"
    extraction_summary = extract_dataset(extraction_input, CONFIG, extracted_path)
    extracted = pd.read_csv(extracted_path, dtype=str, keep_default_na=False)
    pairs = split_extracted_pairs(extracted)

    historical_extraction_input = OUT_DIR / "wos_historical_extraction_input.csv"
    wos_historical[["id", "doi", "display_name", "publication_date", "abstract", "author_keywords", "keywords_plus", "work_type", "source_name"]].to_csv(historical_extraction_input, index=False)
    historical_extracted_path = OUT_DIR / "wos_historical_extracted.csv"
    historical_extraction_summary = extract_dataset(
        historical_extraction_input, CONFIG, historical_extracted_path
    )
    historical_extracted = pd.read_csv(
        historical_extracted_path, dtype=str, keep_default_na=False
    )
    historical_pairs = split_extracted_pairs(historical_extracted)
    historical_pairs.to_csv(OUT_DIR / "wos_historical_extracted_pairs.csv", index=False)
    locked = pd.read_csv(LOCKED_RANKING, dtype=str, keep_default_na=False)
    locked["rank"] = pd.to_numeric(locked["rank"], errors="coerce").astype("Int64")
    locked_key = locked[["system_id", "reaction_id", "rank"]].drop_duplicates()
    if pairs.empty:
        pair_key = pd.DataFrame()
        review = pd.DataFrame()
    else:
        pair_key = pairs.merge(locked_key, on=["system_id", "reaction_id"], how="left")
        pair_key["is_locked_candidate"] = pair_key["rank"].notna()
        pair_key["doi_norm"] = pair_key["doi"].map(normalize_doi)
        pair_key["title_norm"] = pair_key["display_name"].map(normalize_title)
        outcome_match_map = outcome_overlap.set_index("id")["match_method"].to_dict()
        pair_key["openalex_work_match_method"] = pair_key["paper_id"].map(outcome_match_map).fillna("unmatched")
        pair_key = pair_key.sort_values(["is_locked_candidate", "rank", "publication_date"], ascending=[False, True, True], na_position="last").reset_index(drop=True)
        pair_key["blind_id"] = [f"WOS26-{i:03d}" for i in range(1, len(pair_key) + 1)]
        review_columns = [
            "blind_id", "paper_id", "doi", "publication_date", "display_name",
            "composition_raw", "system_id", "reaction_id", "material_class",
            "evidence_context", "formula_in_title", "reaction_in_title",
        ]
        review = pair_key[review_columns].copy()
        review = review.rename(columns={"paper_id": "wos_record_id"})
        for column in [
            "metallic_hea_valid_yes_no_uncertain",
            "target_catalyst_yes_no_uncertain",
            "reaction_valid_yes_no_uncertain",
            "corrected_system_id",
            "corrected_reaction_id",
            "first_report_in_wos_yes_no_uncertain",
            "evidence_quote_or_pointer",
            "reviewer_id",
            "review_status",
            "notes",
        ]:
            review[column] = ""
    pair_key.to_csv(OUT_DIR / "wos_outcome_candidate_key_private.csv", index=False)
    review.to_csv(OUT_DIR / "wos_outcome_blinded_review_template.csv", index=False)

    evaluation_start = pd.Timestamp("2026-01-01")
    evaluation_end = pd.Timestamp("2026-09-16")
    if pair_key.empty:
        focused_key = pair_key.copy()
        focused_review = review.copy()
    else:
        pair_dates = pd.to_datetime(pair_key["publication_date"], errors="coerce")
        focused_key = pair_key[
            pair_key["is_locked_candidate"]
            & pair_dates.between(evaluation_start, evaluation_end, inclusive="both")
        ].copy()
        focused_ids = set(focused_key["blind_id"])
        focused_review = review[review["blind_id"].isin(focused_ids)].copy()
    focused_key.to_csv(OUT_DIR / "wos_locked_candidate_review_key_private.csv", index=False)
    focused_review.to_csv(OUT_DIR / "wos_locked_candidate_blinded_review.csv", index=False)

    if focused_key.empty or historical_pairs.empty:
        potential_prior = pd.DataFrame()
    else:
        focus_pairs = focused_key[["system_id", "reaction_id"]].drop_duplicates()
        potential_prior = historical_pairs.merge(
            focus_pairs, on=["system_id", "reaction_id"], how="inner"
        ).drop_duplicates(subset=["paper_id", "system_id", "reaction_id"])
        potential_prior = potential_prior.sort_values(
            ["system_id", "reaction_id", "publication_date", "paper_id"]
        ).reset_index(drop=True)
        potential_prior["prior_blind_id"] = [
            f"WOSPRE-{i:03d}" for i in range(1, len(potential_prior) + 1)
        ]
        potential_prior = potential_prior[
            [
                "prior_blind_id", "paper_id", "doi", "publication_date", "display_name",
                "composition_raw", "system_id", "reaction_id", "material_class",
                "evidence_context", "formula_in_title", "reaction_in_title",
            ]
        ].rename(columns={"paper_id": "wos_record_id"})
        for column in [
            "same_exact_element_set_yes_no_uncertain",
            "same_target_reaction_yes_no_uncertain",
            "target_catalyst_yes_no_uncertain",
            "qualifying_prior_edge_yes_no_uncertain",
            "evidence_quote_or_pointer",
            "reviewer_id",
            "review_status",
            "notes",
        ]:
            potential_prior[column] = ""
    potential_prior.to_csv(
        OUT_DIR / "wos_potential_pre2026_edges_blinded_review.csv", index=False
    )

    hist_matched = int(historical_overlap["match_method"].ne("unmatched").sum())
    outcome_matched = int(outcome_overlap["match_method"].ne("unmatched").sum())
    event_matched = int(event_coverage["found_in_wos"].sum())
    accepted_matched = int(accepted_coverage["found_in_wos"].sum())
    known_matched = int(known_coverage["found_in_wos"].sum())
    peer_mask = known["peer_reviewed_article"].astype(str).str.lower().eq("true")
    known_peer_coverage = known_coverage.loc[peer_mask]
    peer_matched = int(known_peer_coverage["found_in_wos"].sum())
    event_years = event_coverage[event_coverage["found_in_wos"]].copy()
    event_years["openalex_year"] = pd.to_datetime(event_years["publication_date"], errors="coerce").dt.year
    event_years["wos_year"] = pd.to_datetime(event_years["wos_effective_publication_date"], errors="coerce").dt.year
    comparable_years = event_years.dropna(subset=["openalex_year", "wos_year"])
    exact_year = int((comparable_years["openalex_year"] == comparable_years["wos_year"]).sum())
    within_one_year = int(((comparable_years["openalex_year"] - comparable_years["wos_year"]).abs() <= 1).sum())
    outcome_dates = pd.to_datetime(wos_outcome["publication_date"], errors="coerce")
    outcome_eligible = wos_outcome[outcome_dates.between(evaluation_start, evaluation_end, inclusive="both")]
    outcome_overlap_dates = pd.to_datetime(outcome_overlap["publication_date"], errors="coerce")
    outcome_overlap_eligible = outcome_overlap[outcome_overlap_dates.between(evaluation_start, evaluation_end, inclusive="both")]
    outcome_eligible_matched = int(outcome_overlap_eligible["match_method"].ne("unmatched").sum())
    summary = {
        "status": "automatic_identity_audit_complete_scientific_review_pending",
        "retrieval_date": "2026-10-01",
        "historical_wos_records": int(len(wos_historical)),
        "historical_openalex_records": int(len(oa_historical)),
        "historical_wos_records_matched_to_openalex": hist_matched,
        "historical_wos_to_openalex_overlap": pct(hist_matched, len(wos_historical)),
        "historical_openalex_events": int(len(event_coverage)),
        "historical_openalex_events_found_in_wos": event_matched,
        "historical_event_identity_recall": pct(event_matched, len(event_coverage)),
        "historical_event_year_comparison_rows": int(len(comparable_years)),
        "historical_event_exact_year_agreement": pct(exact_year, len(comparable_years)),
        "historical_event_within_one_year_agreement": pct(within_one_year, len(comparable_years)),
        "outcome_wos_records": int(len(wos_outcome)),
        "outcome_wos_records_in_locked_2026_window": int(len(outcome_eligible)),
        "outcome_openalex_records": int(len(oa_outcome)),
        "outcome_wos_records_matched_to_openalex": outcome_matched,
        "outcome_wos_to_openalex_overlap": pct(outcome_matched, len(wos_outcome)),
        "outcome_window_wos_records_matched_to_openalex": outcome_eligible_matched,
        "outcome_window_wos_to_openalex_overlap": pct(outcome_eligible_matched, len(outcome_overlap_eligible)),
        "previously_accepted_2026_records": int(len(accepted_coverage)),
        "previously_accepted_2026_records_found_in_wos": accepted_matched,
        "known_locked_matches_all_primary_and_preprint": int(len(known_coverage)),
        "known_locked_matches_all_found_in_wos": known_matched,
        "known_peer_reviewed_locked_matches": int(len(known_peer_coverage)),
        "known_peer_reviewed_locked_matches_found_in_wos": peer_matched,
        "wos_extraction": extraction_summary,
        "wos_historical_extraction": historical_extraction_summary,
        "wos_extracted_system_reaction_pairs_for_review": int(len(review)),
        "wos_extracted_pairs_matching_locked_candidates": int(pair_key["is_locked_candidate"].sum()) if not pair_key.empty else 0,
        "locked_window_candidate_rows_awaiting_blinded_review": int(len(focused_review)),
        "machine_detected_potential_pre2026_edge_rows_awaiting_review": int(len(potential_prior)),
        "focused_candidate_pairs_with_potential_pre2026_edge": int(
            potential_prior[["system_id", "reaction_id"]].drop_duplicates().shape[0]
        ) if not potential_prior.empty else 0,
        "scientific_validation_complete": False,
        "interpretation_guardrail": "Database identity overlap is automatic. HEA validity, catalyst role, reaction validity, and first-report status require blinded human review.",
    }
    (OUT_DIR / "automatic_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    readme = f"""# Web of Science independent validation (retrieved 2026-10-01)

The automatic identity audit is complete. It matches records by normalized DOI first and exact normalized title second. It does **not** treat text extraction as scientific confirmation.

- Historical WoS records: {len(wos_historical)}
- Historical WoS records also found in the frozen OpenAlex corpus: {hist_matched}
- Frozen OpenAlex first-report events found by identity in WoS: {event_matched}/{len(event_coverage)}
- 2026 WoS records: {len(wos_outcome)}
- 2026 WoS records also found in the locked OpenAlex outcome snapshot: {outcome_matched}
- Previously accepted 2026 records found in WoS: {accepted_matched}/{len(accepted_coverage)}
- Peer-reviewed locked candidate matches found in WoS: {peer_matched}/{len(known_peer_coverage)}
- Extracted system–reaction rows awaiting blinded review: {len(review)}
- Those rows that privately correspond to frozen candidates: {int(pair_key['is_locked_candidate'].sum()) if not pair_key.empty else 0}
- Focused locked-window rows requiring blinded review: {len(focused_review)}
- Machine-detected possible pre-2026 edges requiring review: {len(potential_prior)}

## Required human step

For the prediction robustness endpoint, fill `wos_locked_candidate_blinded_review.csv` and `wos_potential_pre2026_edges_blinded_review.csv` without opening `wos_locked_candidate_review_key_private.csv`. The focused outcome file contains only records dated from 1 January through 16 September 2026 that machine extraction linked to an element-set–reaction pair in the frozen candidate universe. The historical file checks whether any of those relations were already public before 2026. The private key contains ranks and OpenAlex-overlap information and must remain hidden until review is complete. The 174-row broad template is retained for a more exhaustive corpus audit but is not required for the focused endpoint.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
