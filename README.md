# HEA electrocatalysis temporal forecasting

This repository contains the executable code, public derived data, and numerical
results for chemistry-aware temporal forecasting of high-entropy-alloy (HEA)
electrocatalysis research directions.

## Repository contents

- `src/hea_forecast/`: data collection, extraction, normalization, chemistry
  descriptors, temporal modelling, evaluation, and robustness analysis.
- `configs/`: versioned query and evaluation settings.
- `tests/`: automated unit and temporal-leakage tests.
- `tools/`: analysis, cross-index audit, reliability, forward-portfolio, and
  plotting utilities.
- `data/`: public derived event tables, frozen candidate rankings, performance
  conditions, and de-identified human labels.
- `results/`: temporal metrics, validation statistics, method comparisons,
  chemistry screens, reliability checks, independent-index audit summaries,
  forward predictions, and figures.
- `docs/`: technical protocols, data dictionary, and provenance notes.

No article drafts, tracked revisions, response letters, submission files,
publisher PDFs, full-text articles, licensed database exports, API credentials,
or local working files are included.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m pytest -q
```

## Main reproducible outputs

- Rolling temporal evaluation: `results/temporal/`
- Frozen 2026 validation: `results/validation/`
- Reliability and permutation checks: `results/reliability/`
- Model comparisons: `results/method_comparison/`
- Independent-index audit: `results/wos_audit/`
- Chemistry screen: `results/chemistry/`
- Time-stamped 2027–2028 forward portfolio: `results/forward/`

To rebuild the forward-portfolio selection and its figure from the public audit
table:

```powershell
.\.venv\Scripts\python.exe tools\build_forward_portfolio.py
```

The released forward portfolio contains testable research hypotheses rather than
validated catalytic claims. Absence from the documented literature screens is
not proof of global novelty, synthesizability, phase stability, or catalytic
activity.

## Public-data boundary

The public tables retain structured factual annotations, identifiers, derived
features, scores, and aggregate statistics. Abstract-derived text, evidence
sentences, local PDF filenames, private adjudication keys, and raw licensed
records have been removed. Source records must be obtained separately from the
relevant provider under its terms.

## Integrity and licensing

`MANIFEST_SHA256.csv` provides a SHA-256 checksum for every released file other
than the manifest itself. Software is distributed under GPL-3.0-or-later.
Releaseable derived factual data and numerical results are intended for CC BY
4.0 where no third-party licence applies. See `LICENSE`, `NOTICE`, and
`DATA_LICENSE.md`.

