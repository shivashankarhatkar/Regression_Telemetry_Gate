"""
telemetry/tracing.py
======================

End-to-end tracing instrumentation using Langfuse's OpenTelemetry-native
SDK (v4+) — see week-5-notes.md, section 2 and 2a, for the full
trace/span/generation mental model this implements.

IMPORTANT — current Langfuse SDK version: this uses the OTel-native
`get_client()` / `start_as_current_observation()` API (Langfuse Python
SDK v4). Older tutorials showing `from langfuse.decorators import
observe` and manually-constructed trace objects reflect the pre-OTel
v2 SDK and will not match this module — see
https://langfuse.com/docs/observability/sdk/overview if the API has
moved further since this was written (Langfuse is explicitly a
fast-moving project).

AgentTracer wraps BOTH the real Langfuse client (so traces are visible
in your Langfuse project) AND a local TraceRecorder (usage_aggregator.py)
so cost/latency data is also available synchronously, in-process,
without a network round trip to Langfuse's API — see
usage_aggregator.py's module docstring for why both exist.
"""

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from telemetry.usage_aggregator import TraceRecorder, estimate_cost_usd


class SpanHandle:
    """
    Handle yielded by AgentTracer's context managers, letting node code
    attach output/usage data to the currently-open span before it closes.
    """

    def __init__(self, langfuse_observation: Any) -> None:
        """
        Args:
            langfuse_observation: The Langfuse observation object
                yielded by `start_as_current_observation` (has an
                `.update(...)` method), or None if no Langfuse client
                is configured (see AgentTracer's docstring on graceful
                degradation).
        """
        self._langfuse_observation = langfuse_observation
        self.input_tokens = 0
        self.output_tokens = 0

    def set_usage(self, input_tokens: int, output_tokens: int) -> None:
        """
        Record token usage for a generation span. Call this from inside
        a `tracer.generation(...)` block once the LLM response is available.
        """
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        if self._langfuse_observation is not None:
            self._langfuse_observation.update(
                usage_details={"input": input_tokens, "output": output_tokens}
            )

    def set_output(self, output: Any) -> None:
        """Attach an output value to the currently-open span, visible in Langfuse."""
        if self._langfuse_observation is not None:
            self._langfuse_observation.update(output=output)


class AgentTracer:
    """
    Wraps the Persistent Operator's graph nodes and LLM calls in
    Langfuse spans/generations, while also recording the same
    latency/token/cost data locally via TraceRecorder.

    Designed to degrade gracefully: if Langfuse isn't configured (no
    LANGFUSE_PUBLIC_KEY/SECRET_KEY set), local recording still works —
    useful for running the eval suite (evaluation/) without requiring a
    real Langfuse project, while production use gets full tracing.
    """

    def __init__(self, model: str) -> None:
        """
        Args:
            model: The chat model slug in use, for cost estimation on
                generation spans (see usage_aggregator.estimate_cost_usd).
        """
        self._model = model
        self._recorder = TraceRecorder()
        self._langfuse_client = self._try_get_langfuse_client()

    @staticmethod
    def _try_get_langfuse_client() -> Any | None:
        """
        Attempt to get a configured Langfuse client. Returns None
        (rather than raising) if the langfuse package isn't installed
        or isn't configured — see class docstring on graceful degradation.
        """
        try:
            from langfuse import get_client  # noqa: PLC0415

            return get_client()
        except Exception:
            return None

    @asynccontextmanager
    async def span(self, name: str) -> AsyncIterator[SpanHandle]:
        """
        Trace a plain (non-LLM) step — e.g. a graph node like
        load_context or execute_tools.

        Usage:
            async with tracer.span("execute_tools") as handle:
                ... do the work ...
                handle.set_output(result)
        """
        start = time.monotonic()

        if self._langfuse_client is not None:
            with self._langfuse_client.start_as_current_observation(
                as_type="span", name=name
            ) as observation:
                handle = SpanHandle(observation)
                try:
                    # Yield control back to the caller inside the `with` block, providing the active span handle.
                    # Execution resumes here after the caller's `with` block finishes, allowing the `finally` block to record latency.
                    yield handle
                finally:
                    latency_ms = (time.monotonic() - start) * 1000
                    self._recorder.record_span(name=name, latency_ms=latency_ms)
        else:
            handle = SpanHandle(None)
            try:
                yield handle
            finally:
                latency_ms = (time.monotonic() - start) * 1000
                self._recorder.record_span(name=name, latency_ms=latency_ms)

    @asynccontextmanager
    async def generation(self, name: str) -> AsyncIterator[SpanHandle]:
        """
        Trace an LLM call specifically — uses Langfuse's "generation"
        observation type (see week-5-notes.md section 2), which
        natively understands model/usage data, and additionally records
        estimated cost locally once usage is set via
        `handle.set_usage(...)`.

        Usage:
            async with tracer.generation("call_model") as handle:
                response = await llm.ainvoke(...)
                usage = getattr(response, "usage_metadata", None) or {}
                handle.set_usage(
                    usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                )
                handle.set_output(response.content)
        """
        start = time.monotonic()

        if self._langfuse_client is not None:
            with self._langfuse_client.start_as_current_observation(
                as_type="generation", name=name, model=self._model
            ) as observation:
                handle = SpanHandle(observation)
                try:
                    yield handle
                finally:
                    latency_ms = (time.monotonic() - start) * 1000
                    cost = estimate_cost_usd(self._model, handle.input_tokens, handle.output_tokens)
                    self._recorder.record_span(
                        name=name,
                        latency_ms=latency_ms,
                        input_tokens=handle.input_tokens,
                        output_tokens=handle.output_tokens,
                        cost_usd=cost,
                    )
        else:
            handle = SpanHandle(None)
            try:
                yield handle
            finally:
                latency_ms = (time.monotonic() - start) * 1000
                cost = estimate_cost_usd(self._model, handle.input_tokens, handle.output_tokens)
                self._recorder.record_span(
                    name=name,
                    latency_ms=latency_ms,
                    input_tokens=handle.input_tokens,
                    output_tokens=handle.output_tokens,
                    cost_usd=cost,
                )

    def summary(self):
        """Return the local TraceSummary for everything recorded so far — see usage_aggregator.py."""
        return self._recorder.summary()

    def flush(self) -> None:
        """
        Flush pending Langfuse spans. IMPORTANT in short-lived processes
        (scripts, CI jobs, serverless) — see week-5-notes.md section 2a's
        common mistake about losing unflushed spans on process exit.
        """
        if self._langfuse_client is not None:
            self._langfuse_client.flush()
