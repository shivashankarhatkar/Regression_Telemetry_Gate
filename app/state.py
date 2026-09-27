"""
state.py
========

The LangGraph state schema for the agent's loop — see week-4-notes.md,
section 5, for the full rationale on what belongs in agent state and
why reducer-based incremental updates (rather than full replacement)
are used.

This is deliberately a SEPARATE schema from app/schemas.py's
ChatRequest/ChatResponse: those describe the HTTP API's contract with
callers; this describes what the agent loop itself needs to carry
between iterations. Conflating the two would leak internal
implementation details (like the raw LangChain message objects) into
the public API contract.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ToolCallRecord(TypedDict):
    """
    A record of one tool call, kept in state for two purposes: (1) the
    repeated-identical-call safety valve (week-4-notes.md, section 4)
    compares consecutive records here, and (2) the final API response
    surfaces this list as an inspectable trace (app/schemas.py's
    ToolCallSummary).
    """

    name: str
    arguments: dict


class AgentState(TypedDict):
    """
    The full state carried through one agent loop execution.

    Attributes:
        messages: The conversation history. The `add_messages` reducer
            means node functions return only NEW messages to append,
            not the full history — LangGraph merges them in. This is
            what LangGraph's checkpointer persists per thread_id,
            implementing short-term/session memory (week-4-notes.md,
            section 6) across separate HTTP requests sharing a thread_id.
        customer_id: Which customer this conversation belongs to — used
            to load/save long-term memory (app/memory/customer_memory_store.py),
            independent of thread_id (see app/schemas.py's ChatRequest
            docstring for why these are different identifiers).
        iteration_count: How many times the call_model node has run in
            this execution. Checked against Settings.max_iterations —
            see week-4-notes.md section 4's termination-condition
            discussion. Uses a "last write wins" reducer since each
            call_model invocation sets the authoritative current count,
            rather than something to accumulate via addition.
        tool_call_log: Every tool call made so far, in order. Uses
            operator.add (list concatenation) as its reducer, so each
            node returns only the NEW calls made this step, appended to
            the existing log — same incremental-update pattern as messages.
        customer_context: A rendered summary of this customer's
            long-term memory facts (see app/memory/customer_memory_store.py),
            refreshed by load_context at the start of every request.
            Deliberately kept OUTSIDE of `messages` (not persisted into
            the checkpointed conversation history) — it's injected as a
            system message fresh on every call_model invocation instead
            (see graph.py), so it stays current without accumulating a
            new duplicate system message in history on every turn.
        stop_reason: Set explicitly by call_model or force_stop when a
            safety valve fires ("max_iterations" / "repeated_tool_call"),
            left as "completed" otherwise — read by main.py to populate
            ChatResponse.stopped_reason without fragile string-matching
            on message content.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    customer_id: str
    iteration_count: Annotated[int, lambda _old, new: new]
    tool_call_log: Annotated[list[ToolCallRecord], operator.add]
    customer_context: Annotated[str, lambda _old, new: new]
    stop_reason: Annotated[str, lambda _old, new: new]
