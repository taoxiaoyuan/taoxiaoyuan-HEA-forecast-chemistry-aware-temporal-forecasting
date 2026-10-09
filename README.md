# HEA electrocatalysis temporal forecasting

This repository contains the analysis code, public derived data, and numerical
results for chemistry-aware temporal forecasting of high-entropy-alloy (HEA)
electrocatalysis research directions.

## Contents

- `src/hea_forecast/`: extraction, normalization, chemistry descriptors,
  temporal modelling, evaluation, and audit modules.
- `configs/`: versioned analysis and query configurations.
- `tests/`: automated unit and temporal-leakage tests.
- `tools/`: scientific analysis, robustness, cross-database audit, and plotting
  scripts.
- `data/`: public derived event tables, candidate rankings, performance
  conditions, and de-identified reviewer labels.
- `results/`: temporal metrics, validation statistics, robustness analyses,
  chemistry screening, method comparisons, independent-index audit summaries,
  and generated figures.
- `docs/`: technical protocols, data dictionary, provenance, and
  reproducibility notes.

The repository is limited to executable analysis, derived data, numerical
outputs, technical provenance, and reproducibility metadata.

## Installation and tests

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m pytest -q
```

## Data boundary

Only derived, publication-oriented factual tables and numeric summaries are
included. The release excludes API credentials, raw Web of Science exports,
publisher PDFs, full-text articles, reconstructed abstracts, private review
keys, local caches, and virtual environments. Licensed source records must be
obtained separately from the relevant provider.

The two public ranking tables omit the abstract-derived `text_document` field.
The independent-index folder contains only public adjudication summaries and
aggregate metrics, not the licensed database export.

## Reproducibility

The repository includes a SHA-256 manifest for file-level verification. The
automated test suite contained 36 passing tests when this release was assembled
on 2026-10-09.

## Licensing

Software is distributed under GPL-3.0-or-later. Derived factual annotations and
numeric results are intended for CC BY 4.0 where no third-party license applies.
See `LICENSE`, `NOTICE`, and `DATA_LICENSE.md`.

Upstream provenance is documented in `docs/upstream_provenance.md`.
