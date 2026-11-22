# Aggregate experimental results

These files report existing experiments, not new training or inference.
No record-level clinical arrays, labels, identifiers or weights are included.

| File | Contents |
|---|---|
| `model_summary.csv` | Proposed method: 3 datasets x 3 output pipelines x 19 metrics, estimates/CIs/seed SD |
| `seed_metrics.csv` | Proposed method: 3 datasets x 3 pipelines x 10 seeds |
| `benchmark_event_f1.csv` | Nine methods: Raw/Common Macro Event-F1 and CIs, 54 rows |

The baseline export was prepared on 9 October 2026. This version updates the
six ETCNN benchmark rows to the accepted four-branch, five-seed results
reviewed on 21 November 2026. The old PatchTST-proxy ETCNN rows are not used.
Other event-F1 rows are unchanged.

Boundary and duration errors are reported in seconds. For CTU-UHB, the
corrected time-error estimates, CI limits and seed SDs are half the values
in the October export. This was a unit correction: the predictions, F1,
IoU and event counts did not change.

CIs are record-bootstrap intervals conditional on the available trained seeds;
seed SD describes training variability separately. The tables do not establish
complete historical code identity: in particular, MTU's `faithful_v3` source
binding remains unresolved. This package does not contain all comparator
auxiliary metrics or all comparator per-seed scores.
