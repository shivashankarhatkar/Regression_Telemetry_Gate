"""
evaluation/ragas_faithfulness_metric.py
==========================================

A custom DeepEval metric wrapping RAGAS's Faithfulness scorer — see
week-5-notes.md, sections 7 and 8a, for the concept (does the answer
stay factually consistent with the retrieved context?) and RAGAS's
specific claim-decomposition mechanism.

Why wrap RAGAS inside a DeepEval BaseMetric rather than calling RAGAS's
own `evaluate()` separately: this lets faithfulness sit alongside the
GEval correctness metric and the trajectory metric in ONE
`assert_test(test_case, [metrics...])` call per golden example (see
tests/test_regression.py), so a single pytest/deepeval test run
produces one unified pass/fail per example rather than two separate,
disconnected evaluation passes.

Current RAGAS API used here (verified at time of writing — RAGAS,
like the other tools this week, moves quickly): single-sample scoring
via `ragas.dataset_schema.SingleTurnSample` and
`Faithfulness.single_turn_ascore()`, with the judge LLM wrapped via
`ragas.llms.LangchainLLMWrapper`. A newer `ragas.metrics.collections`
API is emerging in RAGAS's more recent releases with a simpler
`.ascore(user_input=..., response=..., retrieved_contexts=...)` call
shape — check https://docs.ragas.io if this has become the stable
default by the time you run this.
"""

import asyncio

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness


class RagasFaithfulnessMetric(BaseMetric):
    """
    Scores whether an agent's answer is faithful to (factually
    consistent with) its retrieval_context, using RAGAS's Faithfulness
    metric under the hood.

    Only meaningful for golden examples that actually have a
    retrieval_context — see golden_dataset.py's `retrieval_context`
    field and eval_runner.py, which only attaches this metric when
    that field is populated.
    The first evaluation call happens here:
    
        test_regression.py:67
        assert_test(test_case, metrics)
    
        metrics contains:
        ToolTrajectoryMetric()
        build_correctness_metric()
        build_tone_metric()
        RagasFaithfulnessMetric(...)
        DeepEval calls:
        RagasFaithfulnessMetric.measure(test_case)

        Defined in ragas_faithfulness_metric.py:68.

        That method calls:
        asyncio.run(self.a_measure(test_case))
        Then a_measure() manually converts the DeepEval test case into a RAGAS sample:
        sample = SingleTurnSample(
            user_input=test_case.input,
            response=test_case.actual_output,
            retrieved_contexts=test_case.retrieval_context or [],
        )
        Finally, it calls RAGAS:

        self._scorer.single_turn_ascore(sample)
        assert_test(test_case, metrics)
            -> RagasFaithfulnessMetric.measure(test_case)
                -> a_measure(test_case)
                    -> SingleTurnSample(...)
                        -> Faithfulness.single_turn_ascore(sample)
    """

    def __init__(self, judge_llm, threshold: float = 0.7) -> None:
        """
        Args:
            judge_llm: A LangChain-compatible chat model to use as
                RAGAS's evaluator LLM (this project passes the same
                ChatOpenRouter instance used elsewhere — see
                eval_runner.py — since RAGAS's LangchainLLMWrapper
                accepts any LangChain chat model, not a
                provider-specific one).
            threshold: Minimum faithfulness score (0-1) to pass.
        """
        self.threshold = threshold
        self._scorer = Faithfulness(llm=LangchainLLMWrapper(judge_llm))
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Synchronous entry point required by BaseMetric. Delegates to
        the async RAGAS scorer via asyncio.run, since RAGAS's
        single_turn_ascore is async-only.
        """
        return asyncio.run(self.a_measure(test_case))

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """
        Args:
            test_case: Must have `retrieval_context` populated — see
                class docstring. `input` is used as RAGAS's user_input,
                `actual_output` as the response being checked for
                faithfulness to `retrieval_context`.
        """
        sample = SingleTurnSample(
            user_input=test_case.input,
            response=test_case.actual_output,
            retrieved_contexts=test_case.retrieval_context or [],
        )
        self.score = await self._scorer.single_turn_ascore(sample)
        self.success = self.score >= self.threshold
        self.reason = (
            f"Faithfulness score {self.score:.2f} "
            f"({'meets' if self.success else 'below'} threshold {self.threshold})"
        )
        return self.score

    def is_successful(self) -> bool:
        """Required by BaseMetric."""
        return bool(self.success)

    @property
    def __name__(self) -> str:
        """Required by BaseMetric."""
        return "RAGAS Faithfulness"
