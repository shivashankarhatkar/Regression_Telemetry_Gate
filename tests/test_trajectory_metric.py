"""
test_trajectory_metric.py
============================

Unit tests for ToolTrajectoryMetric's deterministic sequence-comparison
logic. No LLM call, no network — see the metric's module docstring on
why this check is deterministic rather than LLM-as-a-Judge.
"""

from deepeval.test_case import LLMTestCase

from evaluation.trajectory_metric import ToolTrajectoryMetric


def _test_case_with_trajectories(expected: list[str], actual: list[str]) -> LLMTestCase:
    """Test helper: build a minimal LLMTestCase carrying only what the metric reads."""
    return LLMTestCase(
        input="irrelevant for this metric",
        actual_output="irrelevant for this metric",
        additional_metadata={"expected_tool_sequence": expected, "actual_tool_sequence": actual},
    )


def test_exact_match_scores_one_and_passes() -> None:
    """An identical expected/actual sequence should score 1.0 and pass."""
    metric = ToolTrajectoryMetric()
    test_case = _test_case_with_trajectories(
        expected=["get_order_status", "issue_refund"], actual=["get_order_status", "issue_refund"]
    )

    score = metric.measure(test_case)

    assert score == 1.0
    assert metric.is_successful() is True


def test_wrong_order_fails() -> None:
    """The same tools called in the wrong order (the exact ordering bug this metric guards against) should fail."""
    metric = ToolTrajectoryMetric()
    test_case = _test_case_with_trajectories(
        expected=["get_order_status", "issue_refund"], actual=["issue_refund", "get_order_status"]
    )

    score = metric.measure(test_case)

    assert score == 0.0
    assert metric.is_successful() is False
    assert "issue_refund" in metric.reason


def test_missing_tool_call_fails() -> None:
    """An agent that skipped an expected tool call entirely should fail."""
    metric = ToolTrajectoryMetric()
    test_case = _test_case_with_trajectories(expected=["get_order_status", "issue_refund"], actual=["get_order_status"])

    assert metric.measure(test_case) == 0.0
    assert metric.is_successful() is False


def test_extra_unexpected_tool_call_fails() -> None:
    """An agent that called an extra, unexpected tool should also fail an exact-match trajectory check."""
    metric = ToolTrajectoryMetric()
    test_case = _test_case_with_trajectories(
        expected=["get_order_status"], actual=["get_order_status", "create_support_ticket"]
    )

    assert metric.measure(test_case) == 0.0


def test_empty_expected_sequence_matches_empty_actual() -> None:
    """A purely conversational turn expecting no tool calls should pass when none occurred."""
    metric = ToolTrajectoryMetric()
    test_case = _test_case_with_trajectories(expected=[], actual=[])

    assert metric.measure(test_case) == 1.0
    assert metric.is_successful() is True
