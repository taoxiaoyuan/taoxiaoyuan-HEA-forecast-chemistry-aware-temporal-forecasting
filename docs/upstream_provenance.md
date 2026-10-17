# Upstream provenance

## Baseline repository

- Repository: `https://github.com/aimat-lab/materials_concepts`
- Branch at acquisition: `development`
- Pinned commit: `39d764a7489a9abe838524db96685b989b47430b`
- Commit timestamp: `2026-07-17T14:02:47+02:00`
- Local location: `upstream/materials_concepts`

The local checkout is treated as immutable and is excluded from the parent Git
repository. Any adapted implementation belongs under `src/hea_forecast` with a
clear modification record.

## Data record

- DOI: `10.6084/m9.figshare.29315819`
- DataCite title: *Data for "Predicting New Research Directions in Materials
  Science Using LLMs and Concept Graphs"*
- Reported archive size: 2,915,742,332 bytes
- Reported contents: OpenAlex works, Llama-2 concepts, graph, semantic and
  topological features, models, and train/validation/test data.
- DataCite rights: `GPL 3.0+`

## Licensing note

The Nature article and DataCite record state GPL 3.0+, while the checked-out
repository's `pyproject.toml` declares MIT and the branch contains no standalone
license file. Until the authors clarify the discrepancy, this project follows
the more conservative GPL-3.0-or-later terms for distributed derivative code,
retains attribution, and documents modifications.

## Bundled demonstration corpus

The repository includes a 5,000-record `materials-science.llama2.works.csv`
demonstration table. It is suitable for pipeline testing but not for study-level
HEA coverage. The initial literal audit found 64 HEA mentions and no records that
also matched the four configured electrocatalytic reactions.

