"""
test_tools.py
==============

Unit tests for app/tools.py. No network calls — these are pure
functions/simulated in-memory state, invoked exactly the way
graph.py's execute_tools node invokes them (via .invoke(args_dict)).

Note on test isolation: _ORDERS and _SUPPORT_TICKETS are module-level
mutable state (deliberately, to simulate a real backend with side
effects — see tools.py's module docstring). Tests that mutate an
order use a dedicated order ID not touched by other tests, to avoid
order-dependent test flakiness.
"""

from app.tools import calculate, create_support_ticket, get_order_status, issue_refund


def test_calculate_basic_arithmetic() -> None:
    """A simple, well-formed expression should evaluate exactly."""
    result = calculate.invoke({"expression": "12 * (3 + 4)"})
    assert result == "84"


def test_calculate_rejects_unsupported_syntax() -> None:
    """
    Anything outside basic arithmetic (e.g. attempting to call a
    function) should fail safely, returning an error string rather
    than executing arbitrary code — see tools.py's calculate docstring
    on why eval() is deliberately not used.
    """
    result = calculate.invoke({"expression": "__import__('os').system('echo hi')"})
    assert "Could not evaluate" in result


def test_get_order_status_known_order() -> None:
    """A known order ID should return its status and total."""
    result = get_order_status.invoke({"order_id": "ORD-1001"})
    assert "delivered" in result
    assert "89.99" in result


def test_get_order_status_unknown_order() -> None:
    """An unknown order ID should return a clear not-found message, not raise."""
    result = get_order_status.invoke({"order_id": "ORD-9999"})
    assert "No order found" in result


def test_issue_refund_succeeds_once() -> None:
    """Issuing a refund should succeed and mark the order as refunded."""
    result = issue_refund.invoke({"order_id": "ORD-1002", "reason": "item damaged"})
    assert "Refund" in result
    assert "34.50" in result

    # Confirm the side effect is visible to a subsequent lookup.
    status = get_order_status.invoke({"order_id": "ORD-1002"})
    assert "refunded=True" in status


def test_issue_refund_twice_is_rejected() -> None:
    """A second refund attempt on an already-refunded order should be rejected, not double-refund."""
    issue_refund.invoke({"order_id": "ORD-1003", "reason": "first request"})
    second_attempt = issue_refund.invoke({"order_id": "ORD-1003", "reason": "second request"})
    assert "already been refunded" in second_attempt


def test_issue_refund_unknown_order() -> None:
    """Refunding a nonexistent order should fail safely with a clear message."""
    result = issue_refund.invoke({"order_id": "ORD-DOES-NOT-EXIST", "reason": "test"})
    assert "no order found" in result.lower()


def test_create_support_ticket_returns_unique_ids() -> None:
    """Each created ticket should get a distinct, formatted ticket ID."""
    first = create_support_ticket.invoke({"summary": "Issue A", "priority": "high"})
    second = create_support_ticket.invoke({"summary": "Issue B", "priority": "low"})
    assert first != second
    assert "TCK-" in first and "TCK-" in second
