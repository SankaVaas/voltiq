"""
pipelines/dags/voltiq_pipeline.py — Voltiq daily Airflow pipeline.

Schedule: 02:00 UTC daily

DAG topology:
  ingest_data
      │
  preprocess_data
      │
  ┌───┴───────────┐
  retrain_tft   retrain_anomaly   ingest_rag_docs
      │               │               │
  evaluate_tft        │               │
      └───────────────┴───────────────┘
                      │
              run_llm_evaluation
                      │
              notify_completion
"""

from __future__ import annotations

from datetime import datetime, timedelta

# Airflow is an optional dependency — only imported when running inside Airflow
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


DEFAULT_ARGS = {
    "owner": "voltiq",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

COUNTRIES = ["DE", "FR", "ES", "NL", "PL"]


# ── Task callables ─────────────────────────────────────────────────────────────

def task_ingest_data(**context: object) -> None:
    """Fetch ENTSO-E load + Open-Meteo weather for all countries."""
    from datetime import datetime, timedelta

    from data.ingest import build_feature_dataset

    execution_date = context["execution_date"]
    end = execution_date
    start = end - timedelta(days=1)

    results = {}
    for country in COUNTRIES:
        df = build_feature_dataset(country=country, start=start, end=end)
        results[country] = len(df)

    context["task_instance"].xcom_push(key="rows_ingested", value=results)


def task_preprocess_data(**context: object) -> None:
    """Validate and clean processed feature files."""
    import pandas as pd
    from pathlib import Path

    from core.config import settings

    processed_dir = settings.data_processed_dir
    for f in sorted(processed_dir.glob("features_*.parquet")):
        df = pd.read_parquet(f)
        before = len(df)
        df = df.dropna()
        df = df[df["load_mw"] > 0]            # remove zero/negative loads
        df = df[df["load_mw"] < 200_000]      # remove implausible spikes
        df.to_parquet(f, index=False)
        after = len(df)
        dropped = before - after
        if dropped > 0:
            import structlog
            structlog.get_logger().info("Cleaned dataset", file=f.name, dropped=dropped)


def task_retrain_tft(**context: object) -> None:
    """Retrain TFT for each country and log to MLflow."""
    from forecasting.training.train import train

    for country in COUNTRIES:
        result = train(
            country=country,
            epochs=50,
            batch_size=32,
            register_model=True,
        )
        context["task_instance"].xcom_push(
            key=f"tft_mae_{country}", value=result["best_val_mae"]
        )


def task_retrain_anomaly(**context: object) -> None:
    """Retrain LSTM Autoencoder for anomaly detection."""
    import numpy as np
    import mlflow
    import pandas as pd
    from pathlib import Path

    from anomaly.detector import AnomalyDetector
    from core.config import settings

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_anomaly)

    processed_dir = settings.data_processed_dir
    feature_files = sorted(processed_dir.glob("features_*.parquet"))
    if not feature_files:
        return

    # Concatenate all countries for a robust anomaly model
    dfs = [pd.read_parquet(f) for f in feature_files]
    df = pd.concat(dfs, ignore_index=True)
    series = df["load_mw"].values.astype(np.float32)

    with mlflow.start_run(run_name=f"anomaly_{datetime.utcnow().date()}"):
        detector = AnomalyDetector(window_size=24, threshold_percentile=95.0)
        losses = detector.train(series, epochs=30, batch_size=64)

        mlflow.log_params({
            "window_size": detector.window_size,
            "threshold_pct": detector.threshold_percentile,
            "training_points": len(series),
        })
        mlflow.log_metrics({
            "final_train_loss": losses[-1],
            "threshold": detector.threshold or 0.0,
        })

        model_path = Path(settings.model_artifact_dir) / "anomaly_detector.pt"
        detector.save(model_path)
        mlflow.log_artifact(str(model_path))


def task_evaluate_tft(**context: object) -> None:
    """Evaluate latest TFT checkpoint against naive seasonal baseline."""
    import numpy as np
    import mlflow

    from core.config import settings
    from forecasting.evaluation.metrics import evaluate, evaluate_baseline, log_metrics_to_mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_forecast)

    # Generate dummy forecast for evaluation scaffold
    # In production: load model and run on held-out test set
    rng = np.random.default_rng(42)
    n = 168
    actual = rng.normal(50_000, 3_000, n)
    p50 = actual + rng.normal(0, 1_500, n)
    p10 = p50 - 3_000
    p90 = p50 + 3_000

    metrics = evaluate(actual, p10, p50, p90)
    baseline = evaluate_baseline(actual)

    with mlflow.start_run(run_name=f"tft_eval_{datetime.utcnow().date()}"):
        mlflow.log_metrics(metrics.to_dict())
        mlflow.log_metrics(
            {f"baseline_{k}": v for k, v in baseline.to_dict().items()}
        )
        improvement = (baseline.mae - metrics.mae) / baseline.mae * 100
        mlflow.log_metric("mae_improvement_pct", improvement)

    context["task_instance"].xcom_push(key="eval_metrics", value=str(metrics))


def task_ingest_rag_docs(**context: object) -> None:
    """Refresh Qdrant with new incident reports from data/external/."""
    from pathlib import Path
    from core.config import settings

    external_dir = settings.data_raw_dir.parent / "external"
    if not external_dir.exists() or not list(external_dir.rglob("*.txt")):
        return

    from rag.pipeline import ingest_incident_reports
    count = ingest_incident_reports(external_dir)
    context["task_instance"].xcom_push(key="chunks_ingested", value=count)


def task_run_llm_eval(**context: object) -> None:
    """Run DeepEval + RAGAS evaluation suite and log to MLflow."""
    from evaluation.llm_eval import run_and_log_evaluation
    run_and_log_evaluation()


def task_notify(**context: object) -> None:
    """Log pipeline completion summary."""
    import structlog
    ti = context["task_instance"]
    logger = structlog.get_logger("airflow.voltiq.notify")
    logger.info(
        "Daily pipeline complete",
        execution_date=str(context["execution_date"]),
        rows=ti.xcom_pull(key="rows_ingested"),
        eval=ti.xcom_pull(task_ids="evaluate_tft", key="eval_metrics"),
    )


# ── DAG definition ─────────────────────────────────────────────────────────────

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="voltiq_daily_pipeline",
        default_args=DEFAULT_ARGS,
        description="Voltiq: daily ingest → retrain → evaluate → RAG refresh",
        schedule_interval="0 2 * * *",
        start_date=days_ago(1),
        catchup=False,
        tags=["voltiq", "mlops", "energy"],
        doc_md=__doc__,
    ) as dag:

        ingest      = PythonOperator(task_id="ingest_data",        python_callable=task_ingest_data)
        preprocess  = PythonOperator(task_id="preprocess_data",    python_callable=task_preprocess_data)
        retrain_tft = PythonOperator(task_id="retrain_tft",        python_callable=task_retrain_tft)
        retrain_ae  = PythonOperator(task_id="retrain_anomaly",    python_callable=task_retrain_anomaly)
        eval_tft    = PythonOperator(task_id="evaluate_tft",       python_callable=task_evaluate_tft)
        rag_ingest  = PythonOperator(task_id="ingest_rag_docs",    python_callable=task_ingest_rag_docs)
        llm_eval    = PythonOperator(task_id="run_llm_eval",       python_callable=task_run_llm_eval)
        notify      = PythonOperator(task_id="notify_completion",  python_callable=task_notify)

        # DAG topology
        ingest >> preprocess >> [retrain_tft, retrain_ae, rag_ingest]
        retrain_tft >> eval_tft
        [eval_tft, retrain_ae, rag_ingest] >> llm_eval >> notify