"""
test_graph_routing.py
=======================

Unit tests for app/graph.py's route_after_model — the conditional edge
deciding whether to execute more tools, force-stop, or end. Tested
against hand-constructed state dicts and fake message objects, with no
LLM call, graph compilation, or Redis involved (see route_after_model's
docstring for why it's structured to allow this).
"""

from langgraph.graph import END

from app.graph import route_after_model


class _FakeMessage:
    """Minimal stand-in for a LangChain AIMessage — only needs .tool_calls."""

    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


def test_routes_to_end_when_no_tool_calls() -> None:
    """A final text answer (no tool calls) should route to END."""
    state = {"messages": [_FakeMessage(tool_calls=[])], "tool_call_log": []}
    assert route_after_model(state) == END


def test_routes_to_execute_tools_when_tool_calls_present() -> None:
    """A response requesting a tool call should route to execute_tools."""
    state = {
        "messages": [_FakeMessage(tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}}])],
        "tool_call_log": [],
    }
    assert route_after_model(state) == "execute_tools"


def test_routes_to_force_stop_on_repeated_identical_tool_call() -> None:
    """Detecting a stuck loop (via tool_call_log) should override to force_stop, not execute_tools."""
    state = {
        "messages": [_FakeMessage(tool_calls=[{"name": "get_order_status", "args": {"order_id": "X"}}])],
        "tool_call_log": [
            {"name": "get_order_status", "arguments": {"order_id": "X"}},
            {"name": "get_order_status", "arguments": {"order_id": "X"}},
        ],
    }
    assert route_after_model(state) == "force_stop"


def test_new_distinct_tool_call_does_not_trigger_force_stop() -> None:
    """A tool call log with no repeated pair should route normally to execute_tools."""
    state = {
        "messages": [_FakeMessage(tool_calls=[{"name": "issue_refund", "args": {"order_id": "X", "reason": "r"}}])],
        "tool_call_log": [
            {"name": "get_order_status", "arguments": {"order_id": "X"}},
        ],
    }
    assert route_after_model(state) == "execute_tools"
