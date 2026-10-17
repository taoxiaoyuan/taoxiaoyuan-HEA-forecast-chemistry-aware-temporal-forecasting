# Study protocol

## Research question

Given literature available up to year `t`, can topology, publication dynamics,
and composition-derived chemical information rank known HEA systems that will be
reported for a previously unlinked electrocatalytic reaction during the next
three years?

## Unit of prediction

The primary unit is `(system_id, reaction_id, cutoff_year)`. `system_id` is the
alphabetically normalized set of constituent metallic elements. Stoichiometric
and material-level identifiers are retained when evidence permits, but they are
not required for the primary analysis.

## Eligibility

A record is eligible when the studied material is the target catalyst, meets the
configured HEA definition, has an identifiable reaction, and has a resolvable
publication date. Reviews and records that merely mention an HEA as background
are excluded from edge creation.

## Temporal event definition

An event is the earliest verified public report connecting a normalized HEA
system to a reaction. Online-first dates take precedence when reliably present.
Linked preprint and journal records represent one discovery event.

At cutoff `t`, candidates must satisfy:

1. The composition node was observed no later than `t`.
2. The reaction node was observed no later than `t`.
3. The composition-reaction edge was not observed no later than `t`.

An eligible candidate is positive if its first verified edge occurs within the
configured future horizon. Recent incomplete years are not treated as complete
negative follow-up.

## Leakage controls

- Rebuild each graph snapshot using only records available at the cutoff.
- Fit imputers, scalers, vocabularies, and encoders on training history only.
- Do not use future citations or later metadata corrections as cutoff features.
- Resolve duplicate and version-linked publications before assigning first dates.
- Evaluate ranking on the complete eligible test candidate universe.

## Planned model ladder

1. Random and reaction/composition popularity baselines.
2. Logistic regression using publication-trend features.
3. Topological heuristics and an upstream-compatible network baseline.
4. Chemistry-only gradient boosting.
5. Network + trend + chemistry gradient boosting.
6. GraphSAGE only after the preceding baselines and quality gates pass.

## Training regimes

Two complementary regimes are retained.

1. **Past-emergence supervision:** train on links that emerged in an earlier
   horizon and evaluate links emerging after the focal cutoff. This most closely
   matches the forecast target but requires a mature literature history.
2. **Historical-link reconstruction:** at each focal cutoff, learn from links
   already observed and use the fitted model only to rank links absent at that
   cutoff. For positive training examples, the target edge is removed from node
   degree and reaction-popularity features. This is compatible with the broad
   link-prediction framing of the upstream baseline and supplies enough HEA
   training examples in this young field.

The second regime is the practical primary analysis if verified
past-emergence positives remain below the predeclared threshold. Results from
the two regimes must be labeled separately and may not be pooled.

## Primary metrics

PR-AUC, Precision@K, Recall@K, NDCG@K, MRR, enrichment over random, and
calibration. ROC-AUC is reported as a secondary comparison metric.
