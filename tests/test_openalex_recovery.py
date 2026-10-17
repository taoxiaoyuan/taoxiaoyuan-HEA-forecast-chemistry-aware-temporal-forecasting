import json
from pathlib import Path

import pandas as pd

from hea_forecast.openalex import recover_partial_jsonl


def test_recover_partial_jsonl_deduplicates_and_skips_bad_tail(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    output = tmp_path / "works.csv"
    item = {
        "matched_query": "q1",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "work": {
            "id": "https://openalex.org/W1",
            "display_name": "Example",
            "publication_date": "2020-01-01",
        },
    }
    source.write_text(json.dumps(item) + "\n" + json.dumps(item) + "\n{", encoding="utf-8")
    summary = recover_partial_jsonl(source, output)
    assert summary["retrieved_rows"] == 2
    assert summary["unique_works"] == 1
    assert summary["malformed_lines"] == 1
    assert len(pd.read_csv(output)) == 1
