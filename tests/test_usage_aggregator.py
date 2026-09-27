"""
test_usage_aggregator.py
===========================

Unit tests for TraceRecorder/TraceSummary and estimate_cost_usd. Pure
arithmetic — no network, no Langfuse, no LLM calls.
"""

import pytest

from telemetry.usage_aggregator import TraceRecorder, estimate_cost_usd


def test_estimate_cost_usd_known_model() -> None:
    """A known model's cost should be computed from its listed input/output prices."""
    cost = estimate_cost_usd("openai/gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(0.15 + 0.60)


def test_estimate_cost_usd_unknown_model_uses_fallback() -> None:
    """An unlisted model should use the fallback price rather than raising."""
    cost = estimate_cost_usd("some/unlisted-model", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost > 0


def test_recorder_summary_aggregates_across_spans() -> None:
    """summary() should sum latency/tokens/cost across every recorded span."""
    recorder = TraceRecorder()
    recorder.record_span("load_context", latency_ms=10.0)
    recorder.record_span("call_model", latency_ms=500.0, input_tokens=100, output_tokens=50, cost_usd=0.002)
    recorder.record_span("execute_tools", latency_ms=25.0)

    summary = recorder.summary()

    assert len(summary.spans) == 3
    assert summary.total_latency_ms == pytest.approx(535.0)
    assert summary.total_input_tokens == 100
    assert summary.total_output_tokens == 50
    assert summary.total_cost_usd == pytest.approx(0.002)


def test_recorder_summary_empty_when_nothing_recorded() -> None:
    """A fresh recorder with no spans should summarize to all-zero totals, not raise."""
    recorder = TraceRecorder()

    summary = recorder.summary()

    assert summary.spans == []
    assert summary.total_latency_ms == 0
    assert summary.total_cost_usd == 0
