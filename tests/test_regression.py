"""
tests/test_regression.py
===========================

THE CI/CD REGRESSION GATE — see week-5-notes.md, section 11.

Run with either:
    pytest tests/test_regression.py
    deepeval test run tests/test_regression.py

Every golden example (evaluation/golden_dataset.py) is replayed
through the real agent graph (evaluation/eval_runner.py) and scored
against: a deterministic tool-trajectory check, GEval correctness and
tone (LLM-as-a-Judge), and — for examples with retrieval_context — a
RAGAS-based faithfulness check. `assert_test` raises a normal pytest
assertion failure on any metric below its threshold, so this file
behaves exactly like an ordinary pytest suite and can gate a GitHub
Actions job (see .github/workflows/regression.yml) the same way a
broken unit test would.

Requires OPENROUTER_API_KEY to be set (both the agent under test and
the GEval/RAGAS judges call a real model) — see README.md and
.env.example.
"""

# Evaluation flow

# test_regression.py loops through every item in GOLDEN_DATASET.
# eval_runner.py replays each example through the real compiled agent graph.
# It captures:
# Final answer
# Actual tool-call sequence
# Retrieval context
# Per-example telemetry
# It creates a DeepEval LLMTestCase.
# assert_test(test_case, metrics) runs all applicable metrics.
# Dataset

# Defined in golden_dataset.py:

# simple_order_lookup: checks order lookup and answer correctness.
# refund_requires_lookup_first: checks lookup happens before refund.
# cross_thread_memory_recall: checks long-term memory across separate threads.
# Each example provides an expected answer summary, expected tool sequence, and optionally retrieval context.

# Metrics

# Trajectory metric in trajectory_metric.py
# Deterministically compares expected and actual tool sequences. Threshold: 1.0.

# GEval correctness in geval_metrics.py
# Uses an LLM judge to check whether the answer conveys the expected facts. Threshold: 0.6.

# GEval tone
# Uses an LLM judge to check professional customer-support tone. Threshold: 0.6.

# RAGAS faithfulness in ragas_faithfulness_metric.py
# Runs only when the example has retrieval_context. It checks whether the final answer is factually supported by that context. Threshold: 0.7.
# In this dataset, RAGAS runs only for simple_order_lookup.

# When services are called

# The main evaluation entry point is test_regression.py:

# AgentTracer also records latency, tokens, and estimated cost during each replay, but that telemetry is separate from answer scoring.

# Currently, the regression result is not stored in a project file or database.

# The flow is:

# Each example runs in test_regression.py.
# run_golden_example() returns an in-memory EvalResult from eval_runner.py.
# The scored LLMTestCase is held in the local variable:

# assert_test(test_case, metrics) evaluates it and reports pass/fail through the terminal or CI logs.
# The detailed values are temporarily held in:

# result.test_case: answer and evaluation metadata
# result.trace_summary: tokens, latency, and cost
# metric.score, metric.reason, and metric.success: metric results
# There is no json.dump(), database write, or report-file generation in the project. GitHub Actions only shows the result in the workflow logs through regression.yml.




import pytest
from deepeval import assert_test

from app.config import get_settings
from evaluation.eval_runner import run_golden_example
from evaluation.geval_metrics import build_correctness_metric, build_tone_metric
from evaluation.golden_dataset import GOLDEN_DATASET
from evaluation.ragas_faithfulness_metric import RagasFaithfulnessMetric
from evaluation.trajectory_metric import ToolTrajectoryMetric


@pytest.mark.parametrize(
    "example", GOLDEN_DATASET, ids=[example.example_id for example in GOLDEN_DATASET]
)
async def test_golden_example_regression(example) -> None:
    """
    One test per golden example — this is the parametrization that
    gives a segmented (not blended) CI report per week-5-notes.md
    section 11's guidance: a failure here names the specific example
    (and, via `example.category` in additional_metadata, the specific
    capability) that regressed, not just "evaluation failed" overall.
    """
    settings = get_settings()

    result = await run_golden_example(example, settings)
    test_case = result.test_case

    metrics = [
        ToolTrajectoryMetric(),
        build_correctness_metric(),
        build_tone_metric(),
    ]

    # Faithfulness only applies to examples that actually have
    # retrieval_context to be faithful TO — see week-5-notes.md
    # section 7 on why correctness doesn't require the same
    # preconditions as faithfulness.
    if example.retrieval_context:
        judge_llm = build_judge_llm(settings)
        metrics.append(RagasFaithfulnessMetric(judge_llm=judge_llm))
    #DeepEval then internally iterates through the metrics and calls each metric with the same automatically.
    assert_test(test_case, metrics)
# For each golden example, your project creates one LLMTestCase in eval_runner.py, then passes that same object to:
# assert_test(test_case, metrics)

# The metrics list contains:
# [
#     ToolTrajectoryMetric(),
#     build_correctness_metric(),
#     build_tone_metric(),
#     RagasFaithfulnessMetric(...),  # only when retrieval_context exists
# ]

# DeepEval internally calls each metric against the same test_case:

# GEval uses the test case fields and makes an LLM judge call.
# RAGAS uses the same fields but performs its own faithfulness judge call.
# Trajectory uses the same object but performs a direct Python comparison.
# The separate files are metric implementations, not separate evaluation test cases:
# geval_metrics.py
# ragas_faithfulness_metric.py
# trajectory_metric.py
# The actual regression test entry point is test_regression.py, which is parameterized once for each item in the golden dataset.

# There is also test_trajectory_metric.py, but that is only a unit test for the trajectory metric itself. It does not run the full agent evaluation pipeline.
# What happens at assert_test
# Your code calls:


# DeepEval internally does approximately:
# for metric in metrics:
#     score = metric.measure(test_case)
#     assert score >= metric.threshold

# For GEval, its internal measure(test_case) then:

# Reads the selected fields:
# test_case.input
# test_case.actual_output
# test_case.expected_output
# Builds a judge prompt using your criteria.
# Calls the judge LLM.
# Parses the judge response into a score and explanation.
# Stores the score/reason internally.

def build_judge_llm(settings):
    """
    Build the LangChain chat model used as RAGAS's evaluator LLM.

    A separate, tiny helper (rather than inlining this) so a future
    change — e.g. using a different, cheaper judge model than the
    agent's own chat model, per week-4-notes.md's model-tiering
    discussion applied to evaluation cost — is a one-line change here,
    not a change scattered through every test.
    """
    from app.llm import build_chat_model

    # NOTE: build_chat_model() returns a model with tools already
    # bound (see app/llm.py) for the AGENT's use — that's harmless for
    # RAGAS's purposes (it only uses the model for plain judging calls,
    # never triggers tool calls), but if this stops being true after a
    # langchain-openrouter update, use a fresh, tool-free ChatOpenRouter
    # instance here instead.
    return build_chat_model(settings)
