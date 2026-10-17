import pandas as pd

from hea_forecast.deduplication import deduplicate_frame, title_fingerprint


def test_title_fingerprint_normalizes_punctuation_and_case() -> None:
    assert title_fingerprint("High-Entropy Alloy!") == title_fingerprint("high entropy alloy")


def test_equal_titles_are_version_linked_with_earliest_date() -> None:
    frame = pd.DataFrame(
        [
            {
                "id": "W-journal",
                "doi": "https://doi.org/10.1/journal",
                "display_name": "An HEA catalyst",
                "publication_date": "2022-02-01",
            },
            {
                "id": "W-preprint",
                "doi": "https://doi.org/10.1/preprint",
                "display_name": "An HEA catalyst",
                "publication_date": "2021-11-01",
            },
        ]
    )
    result = deduplicate_frame(frame)
    assert len(result) == 1
    assert result.loc[0, "publication_date"] == "2021-11-01"
    assert result.loc[0, "version_count"] == 2
