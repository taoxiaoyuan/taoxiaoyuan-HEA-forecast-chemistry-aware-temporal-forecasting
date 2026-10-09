from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT / "results" / "review_2026" / "second_reviewer_completed.csv"
SECOND = ROOT / "results" / "reliability_upgrade_2026" / "second_independent_reviewer_completed.csv"
OUT = ROOT / "results" / "reliability_upgrade_2026"

FIRST_FIELDS = {
    "material_valid": "reviewer2_material_valid",
    "reaction_valid": "reviewer2_reaction_valid",
    "target_catalyst": "reviewer2_target_catalyst",
}
SECOND_FIELDS = {
    "material_valid": "reviewer_material_valid",
    "reaction_valid": "reviewer_reaction_valid",
    "target_catalyst": "reviewer_target_catalyst",
}
ALLOWED = {"yes", "no", "uncertain"}


def clean(value: object) -> str:
    return str(value).strip().lower()


def read_csv_robust(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding).fillna("")
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unable to decode {path}")


def main() -> None:
    first = read_csv_robust(FIRST)
    second = read_csv_robust(SECOND)
    if len(first) != 26 or len(second) != 26:
        raise ValueError(f"Expected 26 rows per reviewer, got {len(first)} and {len(second)}")
    if first["annotation_id"].duplicated().any() or second["annotation_id"].duplicated().any():
        raise ValueError("Duplicate annotation_id detected")
    if set(first["annotation_id"]) != set(second["annotation_id"]):
        raise ValueError("Reviewer annotation_id sets do not match")

    keep_first = ["annotation_id", "blinded_row_id", "doi", "display_name", "system_id", "reaction_id"] + list(FIRST_FIELDS.values())
    keep_second = ["annotation_id"] + list(SECOND_FIELDS.values()) + [
        "reviewer_corrected_system_id", "reviewer_corrected_reaction_id",
        "reviewer_evidence_sentence", "reviewer_notes", "reviewer_status",
    ]
    merged = first[keep_first].merge(second[keep_second], on="annotation_id", validate="one_to_one")

    for field in FIRST_FIELDS.values():
        merged[field] = merged[field].map(clean)
        invalid = sorted(set(merged[field]) - ALLOWED)
        if invalid:
            raise ValueError(f"Invalid first-review labels in {field}: {invalid}")
    for field in SECOND_FIELDS.values():
        merged[field] = merged[field].map(clean)
        invalid = sorted(set(merged[field]) - ALLOWED)
        if invalid:
            raise ValueError(f"Invalid second-review labels in {field}: {invalid}")
    if not merged["reviewer_status"].map(clean).eq("complete").all():
        raise ValueError("Second independent review contains incomplete rows")

    metrics = []
    for label in FIRST_FIELDS:
        a = merged[FIRST_FIELDS[label]]
        b = merged[SECOND_FIELDS[label]]
        metrics.append({
            "decision_level": label,
            "n": len(merged),
            "raw_agreement": float((a == b).mean()),
            "cohen_kappa": float(cohen_kappa_score(a, b, labels=["yes", "no", "uncertain"])),
            "disagreements": int((a != b).sum()),
            "reviewer1_yes": int((a == "yes").sum()),
            "reviewer2_yes": int((b == "yes").sum()),
        })

    merged["reviewer1_final_accept"] = (
        merged[list(FIRST_FIELDS.values())].eq("yes").all(axis=1)
    ).map({True: "accept", False: "not_accept"})
    merged["reviewer2_final_accept"] = (
        merged[list(SECOND_FIELDS.values())].eq("yes").all(axis=1)
    ).map({True: "accept", False: "not_accept"})
    final_a = merged["reviewer1_final_accept"]
    final_b = merged["reviewer2_final_accept"]
    final_agreement = float((final_a == final_b).mean())
    final_kappa = float(cohen_kappa_score(final_a, final_b, labels=["accept", "not_accept"]))
    final_disagreement = final_a != final_b
    metrics.append({
        "decision_level": "final_binary_acceptance",
        "n": len(merged),
        "raw_agreement": final_agreement,
        "cohen_kappa": final_kappa,
        "disagreements": int(final_disagreement.sum()),
        "reviewer1_yes": int((final_a == "accept").sum()),
        "reviewer2_yes": int((final_b == "accept").sum()),
    })

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "human_human_agreement_by_field.csv", index=False)
    merged.to_csv(OUT / "two_reviewer_aligned_labels.csv", index=False)
    public_columns = [
        "annotation_id", "blinded_row_id", "doi", "system_id", "reaction_id",
        *FIRST_FIELDS.values(), *SECOND_FIELDS.values(),
        "reviewer1_final_accept", "reviewer2_final_accept",
    ]
    merged[public_columns].to_csv(OUT / "two_reviewer_labels_public.csv", index=False)

    disagreement_mask = final_disagreement.copy()
    for label in FIRST_FIELDS:
        disagreement_mask |= merged[FIRST_FIELDS[label]] != merged[SECOND_FIELDS[label]]
    disagreements = merged.loc[disagreement_mask].copy()
    disagreements["adjudicated_material_valid"] = ""
    disagreements["adjudicated_reaction_valid"] = ""
    disagreements["adjudicated_target_catalyst"] = ""
    disagreements["adjudicated_final_accept"] = ""
    disagreements["adjudication_evidence"] = ""
    disagreements["adjudication_notes"] = ""
    disagreements["adjudication_status"] = "pending"
    disagreements.to_csv(OUT / "human_human_disagreements_for_adjudication.csv", index=False)

    labels = ["accept", "not_accept"]
    cm = confusion_matrix(final_a, final_b, labels=labels)
    summary = {
        "reviewers": 2,
        "records": len(merged),
        "alignment_key": "annotation_id",
        "reviewer1_source": str(FIRST.relative_to(ROOT)).replace("\\", "/"),
        "reviewer2_source": str(SECOND.relative_to(ROOT)).replace("\\", "/"),
        "final_binary_acceptance": {
            "reviewer1_accepted": int((final_a == "accept").sum()),
            "reviewer2_accepted": int((final_b == "accept").sum()),
            "raw_agreement": final_agreement,
            "cohen_kappa": final_kappa,
            "disagreements": int(final_disagreement.sum()),
            "confusion_matrix_labels": labels,
            "confusion_matrix": cm.tolist(),
        },
        "any_field_disagreement_records": int(disagreement_mask.sum()),
        "adjudication_required": bool(disagreement_mask.any()),
        "note": "Agreement is calculated before adjudication. Original reviewer labels are preserved.",
    }
    (OUT / "human_human_agreement_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(metric_frame.to_string(index=False))


if __name__ == "__main__":
    main()
