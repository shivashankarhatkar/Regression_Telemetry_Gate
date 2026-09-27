"""
evaluation/trajectory_metric.py
==================================

A custom DeepEval metric evaluating an agent's TOOL-CALL TRAJECTORY —
not just its final answer — see week-5-notes.md, section 9, for the
full rationale (an agent can reach a correct answer via an inefficient,
wrong, or accidentally-lucky sequence of tool calls, which final-answer
scoring alone can't detect).

Built as a custom deepeval.metrics.BaseMetric subclass, following
DeepEval's documented custom-metric pattern (measure/a_measure setting
self.score/self.reason/self.success) — see week-5-notes.md section 10.
This is DETERMINISTIC (a direct sequence comparison), not an
LLM-as-a-Judge metric, precisely because "did the agent call these
exact tools in this exact order" is a question code can answer exactly
— see week-5-notes.md section 6's "When Should I NOT Use It" guidance
on LLM-as-a-Judge: don't spend an LLM call on something checkable in code.
"""

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class ToolTrajectoryMetric(BaseMetric):
    """
    Compares the actual sequence of tool names called during an agent
    run against the golden dataset's expected_tool_sequence.

    The actual trajectory is expected to arrive via
    `test_case.additional_metadata["actual_tool_sequence"]` — see
    eval_runner.py, which populates this from the real
    AgentState.tool_call_log captured while replaying a golden example
    through the compiled graph. This is deliberately NOT part of
    DeepEval's built-in LLMTestCase fields, since tool trajectories are
    specific to this project's agent shape.

    The first evaluation call happens here:

    test_regression.py:67
    assert_test(test_case, metrics)

    metrics contains:
    ToolTrajectoryMetric()
    build_correctness_metric()
    build_tone_metric()
    RagasFaithfulnessMetric(...)

    DeepEval then dispatches the same test_case to each metric.

    Trajectory
    DeepEval calls:
    ToolTrajectoryMetric.measure(test_case)

    Defined in trajectory_metric.py:54.
    test_case.additional_metadata["expected_tool_sequence"]
    test_case.additional_metadata["actual_tool_sequence"]
    Then compares the two lists directly.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        """
        Args:
            threshold: Minimum trajectory-match score to pass. Defaults
                to 1.0 (an exact sequence match required) since a
                dependent-tool-call ordering bug (e.g. refunding before
                looking up the order) is exactly the kind of regression
                this metric exists to catch — see week-5-notes.md
                section 9's ordering-correctness check.
        """
        self.threshold = threshold
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Synchronous scoring — pure sequence comparison, no LLM call
        involved (see module docstring on why this is deterministic).
        """
        expected = test_case.additional_metadata.get("expected_tool_sequence", [])
        actual = test_case.additional_metadata.get("actual_tool_sequence", [])

        if expected == actual:
            self.score = 1.0
            self.reason = f"Tool trajectory matched exactly: {actual}"
        else:
            self.score = 0.0
            self.reason = f"Expected tool sequence {expected}, but agent actually called {actual}"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async entry point — delegates to measure() since no I/O is involved."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Required by BaseMetric — returns whether the last measure() call passed threshold."""
        return bool(self.success)

    @property
    def __name__(self) -> str:
        """Required by BaseMetric — the name shown in DeepEval's test output."""
        return "Tool Trajectory Match"
