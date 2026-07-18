# Supervisor routing evaluation corpus

`routing_contract.v1.json` is the versioned, reviewable input set for
`tests/test_eval_routing.py`. It covers the real routing tools, their markers,
the marker parser, and the graph destination after a model has selected a tool.

Run it with:

```bash
uv run pytest tests/test_eval_routing.py
```

The test uses a fake chat model deliberately. It does not call a provider and
must not be reported as live-model routing accuracy. A provider-backed
evaluation needs a separately approved dataset, model/version record, cost
budget, and captured result artifact.

## Evidence record — 2026-07-18

- Corpus schema: `1.0`
- Cases: 9 total — 6 domain handoffs and 3 direct answers
- Failure taxonomy: wrong-domain handoff, missing marker, unregistered transfer
  tool, and unnecessary handoff
- Validation: all 12 routing-contract tests and the complete 306-test suite
  passed on this revision
- Delivery boundary: no external provider call, credential use, deployment,
  or live-model accuracy claim
