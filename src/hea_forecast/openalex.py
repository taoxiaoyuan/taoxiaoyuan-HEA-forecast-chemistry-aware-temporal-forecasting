from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from hea_forecast.config import load_config

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_OQL_URL = "https://api.openalex.org/"
SELECT_FIELDS = (
    "id,doi,display_name,publication_date,type,is_retracted,is_paratext,"
    "abstract_inverted_index,primary_location,authorships,topics,cited_by_count"
)


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct an abstract from OpenAlex's inverted-index representation."""
    if not inverted_index:
        return ""
    positions = [
        (position, token)
        for token, token_positions in inverted_index.items()
        for position in token_positions
    ]
    positions.sort()
    return " ".join(token for _, token in positions)


def canonical_doi(value: str | None) -> str:
    if not value:
        return ""
    doi = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi.removeprefix(prefix)
    return doi.strip()


def _session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "User-Agent": "hea-forecast/0.1 (scientific literature study)",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }
    )
    return session


def _get_json(
    session: requests.Session, url: str, params: dict[str, Any], attempts: int = 5
) -> dict[str, Any]:
    """Retry complete response reads, including truncated chunked responses."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=(15, 90))
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as error:
            response = error.response
            status = response.status_code if response is not None else "unknown"
            detail = (response.text[:500] if response is not None else "").replace("\n", " ")
            last_error = RuntimeError(f"OpenAlex request failed with HTTP {status}: {detail}")
            if status not in {429, 500, 502, 503, 504}:
                break
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
        except requests.RequestException as error:
            last_error = RuntimeError(f"OpenAlex request failed: {type(error).__name__}")
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def iter_works(
    query: str,
    from_date: str,
    to_date: str,
    api_key: str | None,
    max_pages: int | None = None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield OpenAlex works and response metadata using cursor pagination."""
    session = _session()
    cursor: str | None = "*"
    page = 0
    while cursor:
        params: dict[str, Any] = {
            "search": query,
            "filter": (
                f"from_publication_date:{from_date},"
                f"to_publication_date:{to_date},is_retracted:false"
            ),
            "select": SELECT_FIELDS,
            "per_page": 100,
            "cursor": cursor,
        }
        if api_key:
            params["api_key"] = api_key
        payload = _get_json(session, OPENALEX_WORKS_URL, params)
        meta = payload.get("meta", {})
        results = payload.get("results", [])
        for work in results:
            yield work, meta
        page += 1
        if max_pages is not None and page >= max_pages:
            break
        cursor = meta.get("next_cursor")
        if not results:
            break
        time.sleep(0.11)


def iter_oql_works(
    expression: str,
    api_key: str | None,
    max_pages: int | None = None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield OpenAlex works for a precise OQL expression."""
    session = _session()
    cursor: str | None = "*"
    page = 0
    while cursor:
        params: dict[str, Any] = {
            "oql": " ".join(expression.split()),
            "select": SELECT_FIELDS,
            "per_page": 100,
            "cursor": cursor,
        }
        if api_key:
            params["api_key"] = api_key
        payload = _get_json(session, OPENALEX_OQL_URL, params)
        meta = payload.get("meta", {})
        results = payload.get("results", [])
        for work in results:
            if not work.get("is_retracted", False):
                yield work, meta
        page += 1
        if max_pages is not None and page >= max_pages:
            break
        cursor = meta.get("next_cursor")
        if not results:
            break
        time.sleep(0.11)


def _flatten_work(work: dict[str, Any], query: str, retrieved_at: str) -> dict[str, Any]:
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    authorships = work.get("authorships") or []
    topics = work.get("topics") or []
    return {
        "id": work.get("id", ""),
        "doi": canonical_doi(work.get("doi")),
        "display_name": work.get("display_name", ""),
        "publication_date": work.get("publication_date", ""),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "work_type": work.get("type", ""),
        "is_retracted": bool(work.get("is_retracted", False)),
        "is_paratext": bool(work.get("is_paratext", False)),
        "source_id": source.get("id", ""),
        "source_name": source.get("display_name", ""),
        "landing_page_url": primary.get("landing_page_url", ""),
        "cited_by_count": work.get("cited_by_count", 0),
        "author_ids": json.dumps(
            [
                authorship.get("author", {}).get("id", "")
                for authorship in authorships
                if authorship.get("author")
            ],
            ensure_ascii=False,
        ),
        "topics": json.dumps(
            [topic.get("display_name", "") for topic in topics], ensure_ascii=False
        ),
        "matched_query": query,
        "retrieved_at": retrieved_at,
    }


def recover_partial_jsonl(input_jsonl: str | Path, output_csv: str | Path) -> dict[str, Any]:
    """Recover and deduplicate complete records from an interrupted collection."""
    rows: list[dict[str, Any]] = []
    malformed_lines = 0
    with Path(input_jsonl).open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
                rows.append(
                    _flatten_work(
                        item["work"], item.get("matched_query", ""), item.get("retrieved_at", "")
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                malformed_lines += 1
    frame = pd.DataFrame(rows)
    retrieved_rows = len(frame)
    if not frame.empty:
        grouped = frame.groupby("id", sort=False, dropna=False)
        recovered = grouped.first().reset_index()
        queries = grouped["matched_query"].agg(lambda values: " | ".join(sorted(set(values))))
        recovered["matched_query"] = recovered["id"].map(queries)
        recovered = recovered.sort_values(["publication_date", "id"], kind="stable")
    else:
        recovered = frame
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    recovered.to_csv(destination, index=False)
    return {
        "retrieved_rows": retrieved_rows,
        "unique_works": len(recovered),
        "malformed_lines": malformed_lines,
        "output": str(destination.resolve()),
    }


def collect_from_config(
    config_path: str | Path,
    output_dir: str | Path,
    max_pages_per_query: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    collection = config["collection"]
    from_date = str(collection["from_publication_date"])
    to_date = str(collection["to_publication_date"])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    api_key = os.getenv("OPENALEX_API_KEY")

    rows: list[dict[str, Any]] = []
    query_meta: list[dict[str, Any]] = []
    raw_path = destination / "works.raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_stream:
        for query in collection["search_queries"]:
            count_before = len(rows)
            reported_count: int | None = None
            for work, meta in iter_works(
                query=query,
                from_date=from_date,
                to_date=to_date,
                api_key=api_key,
                max_pages=max_pages_per_query,
            ):
                reported_count = meta.get("count")
                raw_stream.write(
                    json.dumps(
                        {"matched_query": query, "retrieved_at": retrieved_at, "work": work},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows.append(_flatten_work(work, query, retrieved_at))
            query_meta.append(
                {
                    "query": query,
                    "reported_count": reported_count,
                    "retrieved_rows": len(rows) - count_before,
                }
            )

    frame = pd.DataFrame(rows)
    retrieved_rows = len(frame)
    if not frame.empty:
        grouped = frame.groupby("id", sort=False, dropna=False)
        deduplicated = grouped.first().reset_index()
        all_queries = grouped["matched_query"].agg(lambda values: " | ".join(sorted(set(values))))
        deduplicated["matched_query"] = deduplicated["id"].map(all_queries)
        deduplicated = deduplicated.sort_values(
            ["publication_date", "id"], kind="stable"
        ).reset_index(drop=True)
    else:
        deduplicated = frame

    csv_path = destination / "works.csv"
    deduplicated.to_csv(csv_path, index=False)
    summary = {
        "retrieved_at": retrieved_at,
        "api_key_used": bool(api_key),
        "from_publication_date": from_date,
        "to_publication_date": to_date,
        "max_pages_per_query": max_pages_per_query,
        "retrieved_rows": retrieved_rows,
        "unique_works": len(deduplicated),
        "queries": query_meta,
        "raw_jsonl_path": str(raw_path.resolve()),
        "csv_path": str(csv_path.resolve()),
    }
    with (destination / "collection_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def collect_oql_from_config(
    config_path: str | Path,
    output_dir: str | Path,
    max_pages_per_query: int | None = None,
) -> dict[str, Any]:
    """Collect precise title/abstract candidates from configured OQL expressions."""
    config = load_config(config_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    api_key = os.getenv("OPENALEX_API_KEY")
    rows: list[dict[str, Any]] = []
    query_meta: list[dict[str, Any]] = []
    raw_path = destination / "works.raw.jsonl"

    with raw_path.open("w", encoding="utf-8") as raw_stream:
        for query_spec in config["collection"]["oql_queries"]:
            name = query_spec["name"]
            expression = query_spec["expression"]
            count_before = len(rows)
            reported_count: int | None = None
            for work, meta in iter_oql_works(
                expression=expression,
                api_key=api_key,
                max_pages=max_pages_per_query,
            ):
                reported_count = meta.get("count")
                raw_stream.write(
                    json.dumps(
                        {"matched_query": name, "retrieved_at": retrieved_at, "work": work},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows.append(_flatten_work(work, name, retrieved_at))
            query_meta.append(
                {
                    "name": name,
                    "expression": " ".join(expression.split()),
                    "reported_count": reported_count,
                    "retrieved_rows": len(rows) - count_before,
                }
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["id"], keep="first").sort_values(
            ["publication_date", "id"], kind="stable"
        )
    csv_path = destination / "works.csv"
    frame.to_csv(csv_path, index=False)
    summary = {
        "retrieved_at": retrieved_at,
        "api_key_used": bool(api_key),
        "max_pages_per_query": max_pages_per_query,
        "retrieved_rows": sum(item["retrieved_rows"] for item in query_meta),
        "unique_works": len(frame),
        "queries": query_meta,
        "raw_jsonl_path": str(raw_path.resolve()),
        "csv_path": str(csv_path.resolve()),
    }
    with (destination / "collection_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary
