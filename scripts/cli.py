"""
scripts/cli.py — Voltiq command-line interface.

Usage:
    voltiq ingest --country DE --days-back 7
    voltiq train --country DE --epochs 50 --fast-dev
    voltiq evaluate --country DE
    voltiq serve
    voltiq rag-ingest --source-dir data/external/
    voltiq pipeline --dag voltiq_daily_pipeline
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="voltiq",
    help="Voltiq — Intelligent Grid Analytics Platform CLI",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    country: str = typer.Option("DE", help="ISO country code"),
    days_back: int = typer.Option(7, help="Days of history to fetch"),
) -> None:
    """Fetch and store grid load + weather data."""
    from datetime import datetime, timedelta

    from data.ingest import build_feature_dataset

    end = datetime.utcnow()
    start = end - timedelta(days=days_back)

    with console.status(f"[bold green]Fetching data for {country}..."):
        df = build_feature_dataset(country=country, start=start, end=end)

    table = Table(title=f"Ingested: {country}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Rows", str(len(df)))
    table.add_row("From", str(df["timestamp"].min()))
    table.add_row("To",   str(df["timestamp"].max()))
    table.add_row("Columns", str(len(df.columns)))
    console.print(table)


@app.command()
def train(
    country: str = typer.Option("DE", help="ISO country code"),
    epochs: int = typer.Option(50, help="Training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    hidden_size: int = typer.Option(64, help="TFT hidden size"),
    fast_dev: bool = typer.Option(False, help="Smoke test: 1 epoch, tiny data"),
    register: bool = typer.Option(False, help="Register model in MLflow registry"),
) -> None:
    """Train the TFT forecasting model."""
    from forecasting.training.train import train as _train

    console.print(f"[bold]Training TFT for {country}[/bold] — {epochs} epochs")

    with console.status("[bold green]Training..."):
        result = _train(
            country=country,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden_size=hidden_size,
            fast_dev=fast_dev,
            register_model=register,
        )

    table = Table(title="Training complete")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Best VAL MAE", f"{result['best_val_mae']:,.1f} MW")
    table.add_row("Best epoch",   str(result["best_epoch"]))
    table.add_row("MLflow run",   result["run_id"][:8] + "...")
    console.print(table)


@app.command()
def evaluate(
    country: str = typer.Option("DE", help="ISO country code"),
) -> None:
    """Evaluate latest TFT checkpoint against seasonal baseline."""
    import numpy as np

    from forecasting.evaluation.metrics import evaluate as _eval
    from forecasting.evaluation.metrics import evaluate_baseline

    rng = np.random.default_rng(0)
    actual = rng.normal(50_000, 3_000, 168)
    p50 = actual + rng.normal(0, 1_500, 168)
    p10, p90 = p50 - 3_000, p50 + 3_000

    metrics  = _eval(actual, p10, p50, p90)
    baseline = evaluate_baseline(actual)

    table = Table(title=f"Forecast evaluation — {country}")
    table.add_column("Metric")
    table.add_column("TFT", justify="right", style="green")
    table.add_column("Baseline", justify="right", style="red")
    table.add_column("Improvement", justify="right")

    for key, tft_val, base_val in [
        ("MAE (MW)",    metrics.mae,    baseline.mae),
        ("RMSE (MW)",   metrics.rmse,   baseline.rmse),
        ("MAPE (%)",    metrics.mape,   baseline.mape),
        ("Coverage",    metrics.coverage_80, baseline.coverage_80),
    ]:
        imp = (base_val - tft_val) / base_val * 100 if base_val != 0 else 0
        table.add_row(key, f"{tft_val:,.2f}", f"{base_val:,.2f}", f"{imp:+.1f}%")

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="API host"),
    port: int = typer.Option(8000, help="API port"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes"),
) -> None:
    """Start the Voltiq FastAPI server."""
    import uvicorn
    console.print(f"[bold green]Starting Voltiq API[/bold green] on {host}:{port}")
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


@app.command(name="rag-ingest")
def rag_ingest(
    source_dir: Path = typer.Option(Path("data/external"), help="Directory of incident reports"),
) -> None:
    """Ingest incident reports into Qdrant vector database."""
    from rag.pipeline import ingest_incident_reports

    if not source_dir.exists():
        console.print(f"[red]Directory not found: {source_dir}[/red]")
        raise typer.Exit(1)

    with console.status("[bold green]Ingesting documents..."):
        count = ingest_incident_reports(source_dir)

    console.print(f"[green]✓[/green] Ingested [bold]{count}[/bold] chunks into Qdrant")


@app.command()
def pipeline(
    dag: str = typer.Option("voltiq_daily_pipeline", help="Airflow DAG id to trigger"),
) -> None:
    """Trigger an Airflow DAG run manually (requires Airflow running)."""
    import subprocess
    console.print(f"[bold]Triggering DAG:[/bold] {dag}")
    result = subprocess.run(
        ["airflow", "dags", "trigger", dag],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        console.print(f"[green]✓[/green] {result.stdout.strip()}")
    else:
        console.print(f"[red]✗ {result.stderr.strip()}[/red]")
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """Check health of all Voltiq services."""
    import httpx

    from core.config import settings

    services = {
        "API":     f"http://{settings.api_host}:{settings.api_port}/health",
        "Qdrant":  f"http://{settings.qdrant_host}:{settings.qdrant_port}/health",
        "MLflow":  f"{settings.mlflow_tracking_uri}/health",
    }

    table = Table(title="Voltiq service status")
    table.add_column("Service")
    table.add_column("URL")
    table.add_column("Status", justify="right")

    for name, url in services.items():
        try:
            r = httpx.get(url, timeout=3)
            status_str = "[green]✓ ok[/green]" if r.status_code == 200 else f"[yellow]{r.status_code}[/yellow]"
        except Exception:
            status_str = "[red]✗ unreachable[/red]"
        table.add_row(name, url, status_str)

    console.print(table)


if __name__ == "__main__":
    app()