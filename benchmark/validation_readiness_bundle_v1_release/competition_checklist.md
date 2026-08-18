# Competition delivery preflight checklist

## Offline evidence checks available now

- [x] Historical evidence hashes and protocol dependencies are indexed.
- [x] Record162/Record160, Snapshot-key, and final-result counts are cross-checked.
- [x] Default current_rules and disabled experimental tie-break state are checked.
- [x] Delivery-fidelity unsupported exits remain explicit.

## External inputs still required before formal validation

- [ ] `full1000_incomplete`: a completed 1000-query run with authoritative terminal records for every planned query
- [ ] `human_precision_missing`: complete independent human labels and adjudication for the frozen blind package
- [ ] `official_scorer_schema_missing`: an exact official scorer or complete official schema and formula

Coverage, stability, source diagnostics, LLM proxy runs, and delivery fidelity must not be substituted for any unchecked item above.
