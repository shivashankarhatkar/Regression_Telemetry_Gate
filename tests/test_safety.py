"""
test_safety.py
================

Unit tests for app/safety.py's hard termination-condition helpers. See
week-4-notes.md, section 4, for why these must exist independently of
the model's own judgment about whether it's done.
"""

from app.safety import has_exceeded_max_iterations, is_repeated_tool_call


def test_has_exceeded_max_iterations_false_when_under_limit() -> None:
    assert has_exceeded_max_iterations(iteration_count=2, max_iterations=5) is False


def test_has_exceeded_max_iterations_true_when_at_limit() -> None:
    """The check is inclusive — hitting the exact limit counts as exceeded."""
    assert has_exceeded_max_iterations(iteration_count=5, max_iterations=5) is True


def test_has_exceeded_max_iterations_true_when_over_limit() -> None:
    assert has_exceeded_max_iterations(iteration_count=9, max_iterations=5) is True


def test_is_repeated_tool_call_false_with_fewer_than_two_calls() -> None:
    """Can't detect a repeat with zero or one recorded calls."""
    assert is_repeated_tool_call([]) is False
    assert is_repeated_tool_call([{"name": "get_order_status", "arguments": {"order_id": "X"}}]) is False


def test_is_repeated_tool_call_true_for_identical_consecutive_calls() -> None:
    """Same tool, same arguments, twice in a row — the stuck-loop signal."""
    log = [
        {"name": "get_order_status", "arguments": {"order_id": "ORD-1001"}},
        {"name": "get_order_status", "arguments": {"order_id": "ORD-1001"}},
    ]
    assert is_repeated_tool_call(log) is True


def test_is_repeated_tool_call_false_for_different_arguments() -> None:
    """Same tool but different arguments is legitimate, not a stuck loop."""
    log = [
        {"name": "get_order_status", "arguments": {"order_id": "ORD-1001"}},
        {"name": "get_order_status", "arguments": {"order_id": "ORD-1002"}},
    ]
    assert is_repeated_tool_call(log) is False


def test_is_repeated_tool_call_only_checks_the_last_two() -> None:
    """An earlier repeat further back in history shouldn't trigger if the most recent call differs."""
    log = [
        {"name": "get_order_status", "arguments": {"order_id": "ORD-1001"}},
        {"name": "get_order_status", "arguments": {"order_id": "ORD-1001"}},
        {"name": "issue_refund", "arguments": {"order_id": "ORD-1001", "reason": "x"}},
    ]
    assert is_repeated_tool_call(log) is False
