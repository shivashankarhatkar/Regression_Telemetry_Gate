"""
evaluation/geval_metrics.py
==============================

GEval (LLM-as-a-Judge) metric construction — see week-5-notes.md,
section 6, for the full GEval mechanism and its documented biases
(position, verbosity, self-preference), which is why criteria below
are written as specific and decomposed as possible rather than a vague
"is this a good answer?" (see that section's Common Mistakes).

Your code:
    creates test_case
    creates metrics
    calls assert_test

DeepEval:
    passes test_case to every metric
    invokes each metric
    handles GEval evaluation internally

Your custom metrics:
    perform their own comparison or RAGAS call inside measure()

GEval
GEval follows the same DeepEval entry point:
assert_test(test_case, metrics)
    -> GEval.measure(test_case)

But GEval’s measure() is implemented inside the DeepEval library, not in your project. It uses the configured LLMTestCaseParams to extract fields from test_case, builds the judge prompt from criteria, and calls the judge LLM.

Your project only configures GEval here:
return GEval(...)
geval_metrics.py:16

In short, the handoff point for all three metrics is:
assert_test(test_case, metrics)

After that, DeepEval calls each metric’s measure(test_case) method.
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


def build_correctness_metric(threshold: float = 0.6) -> GEval:
    """
    Build a GEval metric checking whether the agent's actual answer
    conveys the same substance as the golden example's
    expected_answer_summary — deliberately a SEMANTIC match, not exact
    string equality (week-4/5 notes: LLM output is correctly phrased
    many valid ways).

    Args:
        threshold: Minimum GEval score (0-1) to pass.
    """
# build_correctness_metric() does not evaluate anything. It only creates a configured GEval object:


# The LLMTestCaseParams values are only field selectors. They tell GEval which fields to extract later from the test case.


    return GEval(
        name="Correctness",
        criteria=(
            "Determine whether the 'actual output' conveys the same key facts and "
            "outcome described in the 'expected output', even if worded differently. "
            "Focus on whether the substance (which tools' results were reported, what "
            "action was taken, what information was conveyed) matches — not exact phrasing."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=threshold,
    )


def build_tone_metric(threshold: float = 0.6) -> GEval:
    """
    Build a GEval metric checking the agent maintains an appropriately
    professional, helpful customer-support tone — a dimension a pure
    correctness check wouldn't catch (a factually correct but curt or
    unhelpful-sounding response should still be flagged).

    Args:
        threshold: Minimum GEval score (0-1) to pass.
    """
    return GEval(
        name="Support Tone",
        criteria=(
            "Determine whether the 'actual output' is written in a professional, "
            "helpful, customer-support-appropriate tone — clear, courteous, and not "
            "curt, robotic, or dismissive."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=threshold,
    )
