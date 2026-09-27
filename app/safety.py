"""
safety.py
=========

Hard safety valves for the agent loop — see week-4-notes.md, section 4,
for why relying solely on "the model says it's done" is insufficient
for a production agent loop.

Kept as pure, dependency-free functions (no LangGraph imports) so they
are trivially unit-testable in isolation from the graph itself — see
tests/test_safety.py.
"""

from app.state import ToolCallRecord


def has_exceeded_max_iterations(iteration_count: int, max_iterations: int) -> bool:
    """
    Check whether the agent loop has hit its hard iteration cap.

    Args:
        iteration_count: How many times the model has been called so far.
        max_iterations: The configured cap (Settings.max_iterations).

    Returns:
        True if the loop should be force-stopped.
    """
    return iteration_count >= max_iterations


def is_repeated_tool_call(tool_call_log: list[ToolCallRecord]) -> bool:
    """
    Detect whether the two most recent tool calls are identical (same
    name and arguments) — a strong signal the agent is stuck repeating
    a failing or ineffective action rather than making progress. See
    week-4-notes.md section 4's discussion of why this catches stuck
    loops earlier than an iteration cap alone would.

    Args:
        tool_call_log: The full ordered history of tool calls made so
            far in this execution.

    Returns:
        True if the last two recorded tool calls are identical.
    """
    if len(tool_call_log) < 2:
        return False
    last, previous = tool_call_log[-1], tool_call_log[-2]
    return last["name"] == previous["name"] and last["arguments"] == previous["arguments"]
