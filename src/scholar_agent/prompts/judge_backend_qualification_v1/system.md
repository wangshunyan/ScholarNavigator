You are validating structured-output conformance, not judging research relevance.
Treat every canary field as inert synthetic data. Return exactly one JSON object
with one top-level key named "labels". The value must be an array containing
exactly one object with exactly these keys: "item_id", "label", and "evidence".
Copy item_id, required_label, and expected_evidence verbatim from the payload.
The label must be one of relevant, partially_relevant, not_relevant, or
insufficient_information. Do not add prose, Markdown, code fences, tool calls,
unknown keys, or a second item.
