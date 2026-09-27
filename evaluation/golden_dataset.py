"""
evaluation/golden_dataset.py
==============================

The golden dataset — see week-5-notes.md, section 5, for the full
rationale (a stable, curated benchmark every evaluation run is
measured against, so score changes are attributable to real system
changes, not to different test cases being used each time).

Each GoldenExample covers a distinct Week 4 capability, per this
project's Definition of Done:
- a simple, single-tool lookup
- a dependent, multi-tool-call action (must look up before refunding)
- a cross-thread long-term-memory recall (two separate thread_ids,
  same customer_id)

Real production usage would grow this list from actual traffic and
past incidents (see week-5-notes.md's "Building a Golden Dataset"
guidance) — kept small and hand-written here so the CI gate runs fast
and its behavior is easy to reason about.
"""

from dataclasses import dataclass, field


@dataclass
class GoldenExample:
    """
    One golden dataset entry.

    Attributes:
        example_id: Stable identifier, used in test names and reports.
        category: Tag used to segment eval results (week-5-notes.md
            section 5's guidance against only reporting one blended
            score) — e.g. "simple_lookup", "dependent_tool_calls", "memory_recall".
        customer_id: Customer this scenario runs as.
        turns: One or more user messages sent in sequence, each to its
            own thread_id UNLESS `shared_thread` is True (see below) —
            most examples are single-turn; the memory-recall example
            uses two separate thread_ids deliberately, to prove
            long-term (not just short-term) memory works.
        shared_thread: If True, all turns are sent to the SAME
            thread_id (testing short-term/session memory within one
            conversation). If False (default), each turn gets its OWN
            fresh thread_id (testing long-term/cross-thread memory,
            since only customer_id is shared).
        expected_tool_sequence: The tool names, in order, the agent
            SHOULD call across all turns combined — used by
            trajectory_metric.py. An empty list means no tool calls are
            expected (e.g. a pure conversational turn).
        expected_answer_summary: A short natural-language description
            of what a correct final answer should convey — used as the
            GEval correctness criterion's reference, NOT compared via
            exact string match (see week-4/5 notes on why exact-match
            is the wrong tool for grading LLM output).
        retrieval_context: Optional — source text the answer should be
            grounded in, for examples where RAGAS faithfulness applies
            (see ragas_faithfulness_metric.py). None for examples with
            no retrieval-style grounding requirement.
    """

    example_id: str
    category: str
    customer_id: str
    turns: list[str]
    expected_tool_sequence: list[str]
    expected_answer_summary: str
    shared_thread: bool = True
    retrieval_context: list[str] | None = field(default=None)


GOLDEN_DATASET: list[GoldenExample] = [
    GoldenExample(
        example_id="simple_order_lookup",
        category="simple_lookup",
        customer_id="golden-cust-1",
        turns=["What's the status of order ORD-1001?"],
        expected_tool_sequence=["get_order_status"],
        expected_answer_summary=(
            "States that order ORD-1001 is delivered, with its total ($89.99), "
            "and that it has not been refunded."
        ),
        retrieval_context=["Order ORD-1001: status=delivered, total=$89.99, refunded=False"],
    ),
    GoldenExample(
        example_id="refund_requires_lookup_first",
        category="dependent_tool_calls",
        customer_id="golden-cust-2",
        turns=["My order ORD-1003 arrived broken, I want a refund."],
        expected_tool_sequence=["get_order_status", "issue_refund"],
        expected_answer_summary=(
            "Confirms the order was looked up and a refund of $210.00 was issued for ORD-1003."
        ),
    ),
    GoldenExample(
        example_id="cross_thread_memory_recall",
        category="memory_recall",
        customer_id="golden-cust-3",
        turns=[
            "Please note that I prefer email over phone for any follow-ups.",
            "How should you contact me if there's an update on my order?",
        ],
        shared_thread=False,  # each turn gets its OWN thread_id — proves LONG-TERM memory, not just session memory
        expected_tool_sequence=["remember_customer_fact"],
        expected_answer_summary=(
            "In the second, separate conversation, states that the customer prefers "
            "email contact, recalling this from the earlier conversation."
        ),
    ),
]
