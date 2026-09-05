"""
evaluation/llm_eval.py — LLM evaluation using DeepEval and RAGAS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import mlflow

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalSample:
    question: str
    expected_answer: str
    context_hints: list[str] = field(default_factory=list)


EVAL_DATASET: list[EvalSample] = [
    EvalSample(
        question="What caused the grid frequency deviation in Germany on 2023-07-14?",
        expected_answer=(
            "The frequency deviation was caused by an unexpected trip of a large generation "
            "unit combined with insufficient reserve activation."
        ),
        context_hints=["frequency", "generation", "reserve"],
    ),
    EvalSample(
        question="What is the peak load forecast for France tomorrow?",
        expected_answer=(
            "The peak load forecast for France in the next 24 hours is approximately "
            "65,000 MW occurring in the late afternoon."
        ),
        context_hints=["France", "peak", "forecast"],
    ),
    EvalSample(
        question="How many anomalies were detected in the Dutch grid last week?",
        expected_answer=(
            "Seven anomalous intervals were detected in the Dutch grid last week, "
            "concentrated on Tuesday evening."
        ),
        context_hints=["Netherlands", "anomaly", "intervals"],
    ),
    EvalSample(
        question="What is the current renewable energy share in the Spanish grid?",
        expected_answer=(
            "Renewable energy accounts for approximately 58% of Spain's current generation "
            "mix, led by wind and solar."
        ),
        context_hints=["Spain", "renewable", "wind", "solar"],
    ),
]


def run_deepeval_suite(n_samples: int | None = None) -> dict[str, Any]:
    try:
        from deepeval import evaluate
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            HallucinationMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError:
        logger.warning("deepeval not installed — skipping evaluation")
        return {"status": "skipped", "reason": "deepeval not installed"}

    from rag.pipeline import GridRAGChain

    chain = GridRAGChain()
    samples = EVAL_DATASET[:n_samples] if n_samples else EVAL_DATASET

    test_cases = []
    for sample in samples:
        try:
            result = chain.invoke_with_sources(sample.question)
            retrieved_context = [s["text"] for s in result["sources"]]
            test_cases.append(
                LLMTestCase(
                    input=sample.question,
                    actual_output=result["answer"],
                    expected_output=sample.expected_answer,
                    retrieval_context=retrieved_context,
                    context=retrieved_context,
                )
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to generate answer for eval sample", error=str(e))

    if not test_cases:
        return {"status": "error", "reason": "no test cases generated"}

    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=settings.llm_model),
        FaithfulnessMetric(threshold=0.8, model=settings.llm_model),
        HallucinationMetric(threshold=0.3, model=settings.llm_model),
    ]

    results = evaluate(test_cases=test_cases, metrics=metrics)

    scores: dict[str, Any] = {
        "answer_relevancy": _avg_score(results, "AnswerRelevancy"),
        "faithfulness": _avg_score(results, "Faithfulness"),
        "hallucination_rate": _avg_score(results, "Hallucination"),
        "n_samples": len(test_cases),
        "status": "ok",
    }
    logger.info("DeepEval completed", **{k: v for k, v in scores.items() if k != "status"})
    return scores


def _avg_score(results: Any, metric_name: str) -> float:  # noqa: ANN401
    scores = [
        r.metrics_data.get(metric_name, {}).get("score", 0.0)
        for r in results
        if hasattr(r, "metrics_data")
    ]
    return round(sum(scores) / max(len(scores), 1), 4)


def run_ragas_eval(n_samples: int | None = None) -> dict[str, Any]:
    try:
        from datasets import Dataset as HFDataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError:
        logger.warning("ragas not installed — skipping RAGAS evaluation")
        return {"status": "skipped", "reason": "ragas not installed"}

    from rag.pipeline import GridRAGChain

    chain = GridRAGChain()
    samples = EVAL_DATASET[:n_samples] if n_samples else EVAL_DATASET

    rows: dict[str, list[Any]] = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for sample in samples:
        try:
            result = chain.invoke_with_sources(sample.question)
            rows["question"].append(sample.question)
            rows["answer"].append(result["answer"])
            rows["contexts"].append([s["text"] for s in result["sources"]])
            rows["ground_truth"].append(sample.expected_answer)
        except Exception as e:  # noqa: BLE001
            logger.warning("RAGAS sample failed", error=str(e))

    if not rows["question"]:
        return {"status": "error", "reason": "no samples"}

    dataset = HFDataset.from_dict(rows)
    result = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return {
        "faithfulness": round(result["faithfulness"], 4),
        "answer_relevancy": round(result["answer_relevancy"], 4),
        "context_precision": round(result["context_precision"], 4),
        "status": "ok",
    }


def run_and_log_evaluation() -> None:
    """Run full eval suite and log results to MLflow."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("voltiq_llm_eval")

    with mlflow.start_run(run_name="eval_suite"):
        deepeval_scores = run_deepeval_suite()
        ragas_scores = run_ragas_eval()

        if deepeval_scores.get("status") == "ok":
            mlflow.log_metrics(
                {
                    "deepeval_relevancy": deepeval_scores["answer_relevancy"],
                    "deepeval_faithfulness": deepeval_scores["faithfulness"],
                    "deepeval_hallucination": deepeval_scores["hallucination_rate"],
                }
            )

        if ragas_scores.get("status") == "ok":
            mlflow.log_metrics(
                {
                    "ragas_faithfulness": ragas_scores["faithfulness"],
                    "ragas_relevancy": ragas_scores["answer_relevancy"],
                    "ragas_context_precision": ragas_scores["context_precision"],
                }
            )

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump({"deepeval": deepeval_scores, "ragas": ragas_scores}, tmp, indent=2)
            mlflow.log_artifact(tmp.name)
        logger.info("Evaluation results logged to MLflow")
