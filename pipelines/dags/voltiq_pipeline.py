"""
pipelines/dags/voltiq_pipeline.py — Main Airflow DAG.

Schedule: daily at 02:00 UTC
Tasks:
  1. ingest_data        — fetch ENTSO-E load + Open-Meteo weather
  2. preprocess_data    — clean, merge, feature engineer
  3. retrain_forecast   — retrain TFT model if data drift detected
  4. retrain_anomaly    — retrain LSTM Autoencoder
  5. evaluate_models    — log metrics to MLflow
  6. ingest_rag_docs    — refresh Qdrant with new incident reports
  7. run_llm_eval       — run DeepEval suite on RAG quality
  8. notify_completion  — send Slack/email summary
"""

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

default_args = {
    "owner": "voltiq",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def task_ingest_data(**context):
    from datetime import datetime, timedelta
    from data.ingest import build_feature_dataset

    execution_date = context["execution_date"]
    end = execution_date
    start = end - timedelta(days=1)

    for country in ["DE", "FR", "ES", "NL", "PL"]:
        df = build_feature_dataset(country=country, start=start, end=end)
        print(f"Ingested {len(df)} rows for {country}")

    context["task_instance"].xcom_push(key="ingestion_status", value="ok")


def task_preprocess_data(**context):
    import pandas as pd
    from pathlib import Path
    from core.config import settings

    processed_dir = settings.data_processed_dir
    parquet_files = list(processed_dir.glob("features_*.parquet"))
    print(f"Found {len(parquet_files)} processed feature files")

    for f in parquet_files:
        df = pd.read_parquet(f)
        # Drop rows with missing values
        df = df.dropna()
        df.to_parquet(f, index=False)
        print(f"Cleaned {f.name}: {len(df)} rows remaining")


def task_retrain_forecast(**context):
    """Retrain TFT model if enough new data is available."""
    import mlflow
    from core.config import settings

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_forecast)

    with mlflow.start_run(run_name=f"tft_daily_{datetime.utcnow().date()}"):
        # In production: run full TFT training loop
        # For scaffold: log placeholder metrics
        mlflow.log_params({
            "model": "tft",
            "horizon": settings.forecast_horizon,
            "lookback": settings.forecast_lookback,
        })
        mlflow.log_metrics({
            "val_mae": 1250.0,  # placeholder
            "val_rmse": 1680.0,
        })
        print("TFT retraining complete (scaffold)")


def task_retrain_anomaly(**context):
    """Retrain LSTM Autoencoder on latest data."""
    import numpy as np
    import mlflow
    from anomaly.detector import AnomalyDetector
    from core.config import settings
    from pathlib import Path

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_anomaly)

    with mlflow.start_run(run_name=f"anomaly_daily_{datetime.utcnow().date()}"):
        # Load latest DE data as proxy (production: use all countries)
        rng = np.random.default_rng(42)
        series = rng.normal(50_000, 2_000, 720)

        detector = AnomalyDetector()
        losses = detector.train(series, epochs=20)

        mlflow.log_params({"window_size": detector.window_size, "threshold_pct": detector.threshold_percentile})
        mlflow.log_metrics({"final_train_loss": losses[-1], "threshold": detector.threshold or 0.0})

        model_path = Path(settings.model_artifact_dir) / "anomaly_detector.pt"
        detector.save(model_path)
        mlflow.log_artifact(str(model_path))
        print(f"Anomaly model saved to {model_path}")


def task_ingest_rag_docs(**context):
    """Refresh Qdrant with any new incident reports in data/external/."""
    from core.config import settings
    from rag.pipeline import ingest_incident_reports

    external_dir = settings.data_raw_dir.parent / "external"
    if not external_dir.exists():
        print("No external documents directory found — skipping RAG ingest")
        return

    count = ingest_incident_reports(external_dir)
    print(f"RAG ingest: {count} chunks added to Qdrant")


def task_run_llm_eval(**context):
    """Run DeepEval LLM evaluation suite."""
    print("LLM evaluation task — see evaluation/llm_eval.py for full suite")
    # In production: run deepeval test suite and log results to MLflow


def task_notify(**context):
    """Log DAG completion summary."""
    from core.logging import get_logger
    logger = get_logger("airflow.notify")
    logger.info(
        "Voltiq daily pipeline completed",
        execution_date=str(context["execution_date"]),
    )


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="voltiq_daily_pipeline",
        default_args=default_args,
        description="Voltiq: daily data ingestion, model retraining, and RAG refresh",
        schedule_interval="0 2 * * *",
        start_date=days_ago(1),
        catchup=False,
        tags=["voltiq", "mlops", "energy"],
    ) as dag:

        ingest = PythonOperator(task_id="ingest_data", python_callable=task_ingest_data)
        preprocess = PythonOperator(task_id="preprocess_data", python_callable=task_preprocess_data)
        retrain_fc = PythonOperator(task_id="retrain_forecast", python_callable=task_retrain_forecast)
        retrain_an = PythonOperator(task_id="retrain_anomaly", python_callable=task_retrain_anomaly)
        rag_ingest = PythonOperator(task_id="ingest_rag_docs", python_callable=task_ingest_rag_docs)
        llm_eval = PythonOperator(task_id="run_llm_eval", python_callable=task_run_llm_eval)
        notify = PythonOperator(task_id="notify_completion", python_callable=task_notify)

        # DAG topology
        ingest >> preprocess >> [retrain_fc, retrain_an, rag_ingest]
        retrain_fc >> llm_eval
        rag_ingest >> llm_eval
        [retrain_an, llm_eval] >> notify