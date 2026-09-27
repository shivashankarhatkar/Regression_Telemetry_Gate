"""
evaluation/eval_runner.py
============================

Replays a GoldenExample through the ACTUAL compiled agent graph
(app/graph.py) — not a mock or a simplified stand-in — capturing both
the final answer and the full tool-call trajectory, then packages the
result into a DeepEval LLMTestCase ready for scoring.

Runs HERMETICALLY: uses LangGraph's in-memory checkpointer/store
(InMemorySaver / InMemoryStore) instead of the Redis-backed ones
app/main.py uses in production. This is a deliberate choice for a CI
eval suite — see week-5-notes.md section 11's point that a regression
gate needs to be "fast enough to run on every PR": requiring a live
Redis instance just to run the eval suite would add real setup
friction and flakiness to every CI run, and each golden example is
independent, so no cross-example persistence is actually needed. A
tracer is still attached (see AgentTracer's graceful degradation) so
running evals ALSO produces local unit-economics data per example,
even without Langfuse configured.

NOTE: InMemorySaver/InMemoryStore are LangGraph's current names for
what older tutorials call MemorySaver — see week-4-notes.md section 10's
common mistake about checking current names in a fast-moving framework.
"""

from dataclasses import dataclass

from deepeval.test_case import LLMTestCase
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.config import Settings
from app.graph import build_agent_graph
from app.llm import build_chat_model
from evaluation.golden_dataset import GoldenExample
from telemetry.tracing import AgentTracer
from telemetry.usage_aggregator import TraceSummary


@dataclass
class EvalResult:
    """
    The output of replaying one GoldenExample: a ready-to-score
    LLMTestCase plus the local telemetry captured while producing it.

    Attributes:
        test_case: DeepEval test case — see build_test_case for how its
            fields are populated from the golden example and the
            agent's actual run.
        trace_summary: Cost/latency/token totals for this one example's
            run (see telemetry/usage_aggregator.py) — since a fresh
            AgentTracer is built per example (see run_golden_example),
            this summary is ISOLATED to this example, unlike
            app/main.py's process-cumulative usage.
    """

    test_case: LLMTestCase
    trace_summary: TraceSummary


async def run_golden_example(example: GoldenExample, settings: Settings) -> EvalResult:
    """
    Execute one golden example's turns against a fresh, hermetic agent
    graph instance and package the result for scoring.

    Args:
        example: The golden example to replay.
        settings: Application settings (chat model, max_iterations) —
            reused so the eval harness exercises the SAME configuration
            production actually runs with, not a hardcoded stand-in.

    Returns:
        An EvalResult with a populated LLMTestCase and this example's
        isolated trace summary.
    """
    llm = build_chat_model(settings)
    tracer = AgentTracer(model=settings.chat_model)
    workflow = build_agent_graph(llm, max_iterations=settings.max_iterations, tracer=tracer)
    compiled_graph = workflow.compile(checkpointer=InMemorySaver(), store=InMemoryStore())

    final_result = None
    all_tool_calls: list[str] = []

    for turn_index, message in enumerate(example.turns):
        # shared_thread=True (the default) reuses one thread_id across
        # turns, testing SHORT-TERM/session memory. shared_thread=False
        # gives each turn its own thread_id, testing LONG-TERM/cross-thread
        # memory instead — see golden_dataset.py's field docstring.
        thread_id = (
            f"{example.example_id}-thread"
            if example.shared_thread
            else f"{example.example_id}-thread-{turn_index}"
        )
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": settings.recursion_limit,
        }

        final_result = await compiled_graph.ainvoke(
            {
                "messages": [HumanMessage(content=message)],
                "customer_id": example.customer_id,
                "iteration_count": 0,
                "tool_call_log": [],
                "customer_context": "",
                "stop_reason": "completed",
            },
            config=config,
        )
        all_tool_calls.extend(record["name"] for record in final_result["tool_call_log"])

    tracer.flush()

    test_case = LLMTestCase(
        input=example.turns[-1],
        actual_output=final_result["messages"][-1].content,
        expected_output=example.expected_answer_summary,
        retrieval_context=example.retrieval_context,
        additional_metadata={
            "expected_tool_sequence": example.expected_tool_sequence,
            "actual_tool_sequence": all_tool_calls,
            "category": example.category,
        },
    )

    return EvalResult(test_case=test_case, trace_summary=tracer.summary())
