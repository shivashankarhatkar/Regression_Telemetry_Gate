"""
The Regression & Telemetry Gate
==================================

Week 5 project: an automated eval and tracing harness built on top of
Week 4's Persistent Operator. Two things happen here that didn't
before:

1. TRACING — every agent graph node and LLM call is wrapped in an
   OpenTelemetry-native Langfuse span/generation (see
   telemetry/tracing.py), capturing token usage, cost, and latency per
   step, not just per request.

2. EVALUATION + CI/CD GATING — a curated golden dataset
   (evaluation/golden_dataset.py) is replayed through the actual agent
   graph on every pull request. Each result is scored with DeepEval
   GEval metrics (correctness/tone), a custom trajectory metric
   (did the agent call the right tools in the right order?), and a
   RAGAS-based faithfulness metric for retrieval-grounded answers.
   A failing score fails the CI job — see .github/workflows/regression.yml
   — making a quality regression "physically unable to ship," per this
   week's brief.

See README.md for setup and usage. See week-5-notes.md (provided
separately) for the concepts this project implements.
"""
