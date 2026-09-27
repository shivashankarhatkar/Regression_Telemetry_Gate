"""
telemetry/usage_aggregator.py
================================

Local, synchronous recording of per-span cost/token/latency data — see
week-5-notes.md, section 3 ("Unit Economics").

Why this exists ALONGSIDE Langfuse tracing (tracing.py) rather than
relying on Langfuse alone: Langfuse's trace data lives in its own
backend, queried via its API — fine for a dashboard, but awkward for
(a) a quick in-process cost summary immediately after a request, and
(b) unit tests, which shouldn't need network access to a real Langfuse
project just to verify aggregation math. TraceRecorder is a small,
dependency-free, testable local mirror of the same per-span data that
is ALSO sent to Langfuse — the same relationship Week 1's UsageTracker
had to the LLM provider's own billing dashboard: local, per-request
detail versus a remote, aggregate source of truth.
"""

from dataclasses import dataclass, field


# Placeholder pricing table — same caveat as Week 1's pricing.py: verify
# current figures at https://openrouter.ai/models before relying on
# this for real cost tracking. Duplicated here (rather than importing
# Week 1's ai-gateway project) so this project remains a single,
# self-contained deliverable.
PRICING_USD_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    # model_slug: (input_price, output_price)
    "anthropic/claude-sonnet-4.5": (3.00, 15.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
}
_DEFAULT_PRICE = (3.00, 15.00)  # fallback if an unlisted model is used


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the USD cost of one LLM call.

    Args:
        model: Model slug used for the call.
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens generated.

    Returns:
        Estimated cost in USD, rounded to 6 decimal places (see Week
        1's pricing.py for why 6, not 2, decimal places — per-call
        costs are fractions of a cent).
    """
    input_price, output_price = PRICING_USD_PER_MILLION_TOKENS.get(model, _DEFAULT_PRICE)
    cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
    return round(cost, 6)


@dataclass
class SpanRecord:
    """
    A single recorded span's telemetry — one node execution or one LLM
    generation within a traced agent run.

    Attributes:
        name: The span's name (e.g. "call_model", "execute_tools").
        latency_ms: How long this span took.
        input_tokens: Prompt tokens consumed, if this was a generation span.
        output_tokens: Completion tokens generated, if this was a generation span.
        cost_usd: Estimated cost, if this was a generation span (0.0 for plain spans).
    """

    name: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class TraceSummary:
    """
    Aggregated totals across every span recorded for one traced request.

    Attributes:
        spans: Every individual SpanRecord, in execution order.
        total_latency_ms: Sum of every span's latency (NOT wall-clock
            request time if spans overlap — this project's spans are
            sequential, so the two coincide here, but that's an
            assumption worth stating explicitly rather than silently
            relying on).
        total_input_tokens: Sum of input tokens across all generation spans.
        total_output_tokens: Sum of output tokens across all generation spans.
        total_cost_usd: Sum of cost across all generation spans.
    """

    spans: list[SpanRecord]
    total_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


class TraceRecorder:
    """
    Collects SpanRecords and produces a TraceSummary.

    Used in two different lifecycles in this project, both valid:
    - app/main.py builds ONE AgentTracer (and therefore one
      TraceRecorder) at process startup, so its summary() is a running,
      cumulative total across every request the process has served —
      the same process-lifetime-cumulative pattern as Week 1's
      UsageTracker. Per-REQUEST breakdowns are available in Langfuse
      itself (each request is its own trace there); this local summary
      is the quick, in-process "unit economics so far" view.
    - evaluation/eval_runner.py builds a FRESH AgentTracer per golden
      example specifically to get an ISOLATED summary for that one
      example, since eval runs care about per-example cost/latency,
      not a cumulative total across the whole suite.
    """

    def __init__(self) -> None:
        self._spans: list[SpanRecord] = []

    def record_span(
        self,
        name: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record one completed span's telemetry."""
        self._spans.append(
            SpanRecord(
                name=name,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )
        )

    def summary(self) -> TraceSummary:
        """
        Produce the aggregated TraceSummary for every span recorded so far.

        This is the "latency breakdown per request" and "unit economics"
        artifact from week-5-notes.md, section 3 — e.g. printed or
        logged immediately after a `/chat` request completes.
        """
        return TraceSummary(
            spans=list(self._spans),
            total_latency_ms=sum(s.latency_ms for s in self._spans),
            total_input_tokens=sum(s.input_tokens for s in self._spans),
            total_output_tokens=sum(s.output_tokens for s in self._spans),
            total_cost_usd=round(sum(s.cost_usd for s in self._spans), 6),
        )
