"""
forecasting/training/train.py — Full TFT training loop with MLflow tracking.

Usage (local CPU):
    python -m forecasting.training.train --country DE --epochs 5 --fast-dev

Usage (Colab T4 — see notebooks/train_tft_colab.ipynb):
    python -m forecasting.training.train --country DE --epochs 50

What this does:
  1. Loads feature dataset (real ENTSO-E or synthetic fallback)
  2. Builds sliding-window sequences for encoder/decoder
  3. Trains TFT with quantile loss (p10, p50, p90)
  4. Logs every epoch to MLflow (params, metrics, loss curves)
  5. Saves best checkpoint to data/artifacts/
  6. Registers model in MLflow Model Registry
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from core.config import settings
from core.logging import get_logger
from forecasting.models.tft import TemporalFusionTransformer, TFTConfig

logger = get_logger(__name__)

ARTIFACT_DIR = settings.model_artifact_dir
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ["load_mw", "temperature_2m", "wind_speed_10m",
                "shortwave_radiation", "hour_of_day", "day_of_week", "month"]
CATEGORICAL_COLS = ["country_id", "is_weekend"]
TARGET_COL = "load_mw"


# ── Dataset ───────────────────────────────────────────────────────────────────

class GridSequenceDataset(Dataset):
    """
    Sliding-window dataset for TFT training.

    Each sample:
      enc_numeric   : (encoder_length, num_numeric)
      enc_categorical: (encoder_length, num_categorical)
      dec_numeric   : (decoder_length, num_numeric)  — known future covariates
      dec_categorical: (decoder_length, num_categorical)
      target        : (decoder_length,)              — future load values
    """

    def __init__(
        self,
        df: pd.DataFrame,
        encoder_length: int = 168,
        decoder_length: int = 48,
    ) -> None:
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.window = encoder_length + decoder_length

        # Normalise numerics (zero-mean, unit-variance)
        self.numeric_mean = df[NUMERIC_COLS].mean()
        self.numeric_std = df[NUMERIC_COLS].std().replace(0, 1)
        norm = (df[NUMERIC_COLS] - self.numeric_mean) / self.numeric_std

        self.numeric = torch.tensor(norm.values, dtype=torch.float32)
        self.categorical = torch.tensor(
            df[CATEGORICAL_COLS].values.astype(int), dtype=torch.long
        )
        self.target = torch.tensor(df[TARGET_COL].values, dtype=torch.float32)

        self.n_samples = len(df) - self.window + 1
        assert self.n_samples > 0, f"Dataset too short: {len(df)} rows < window {self.window}"

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc_end = idx + self.encoder_length
        dec_end = enc_end + self.decoder_length
        return {
            "enc_numeric":    self.numeric[idx:enc_end],
            "enc_categorical": self.categorical[idx:enc_end],
            "dec_numeric":    self.numeric[enc_end:dec_end],
            "dec_categorical": self.categorical[enc_end:dec_end],
            "target":         self.target[enc_end:dec_end],
        }


# ── Loss ──────────────────────────────────────────────────────────────────────

def quantile_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    quantiles: list[float],
) -> torch.Tensor:
    """
    Pinball loss for quantile regression.
    preds:  (batch, horizon, num_quantiles)
    target: (batch, horizon)
    """
    target_expanded = target.unsqueeze(-1).expand_as(preds)
    errors = target_expanded - preds
    q_tensor = torch.tensor(quantiles, dtype=torch.float32, device=preds.device)
    loss = torch.max(q_tensor * errors, (q_tensor - 1) * errors)
    return loss.mean()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    preds_p50: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """MAE, RMSE, MAPE on the median (p50) forecast."""
    p50 = preds_p50.detach().cpu()
    tgt = target.detach().cpu()

    mae = (p50 - tgt).abs().mean().item()
    rmse = ((p50 - tgt) ** 2).mean().sqrt().item()
    mape = ((p50 - tgt).abs() / (tgt.abs() + 1e-8)).mean().item() * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    country: str = "DE",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.15,
    hidden_size: int = 64,
    lstm_layers: int = 2,
    attention_heads: int = 4,
    dropout: float = 0.1,
    fast_dev: bool = False,
    register_model: bool = False,
) -> dict[str, Any]:
    """
    Train the TFT model and log everything to MLflow.

    Args:
        country:        ISO country code for training data
        epochs:         number of training epochs
        batch_size:     mini-batch size
        lr:             initial learning rate
        val_split:      fraction of data held out for validation
        hidden_size:    TFT hidden dimension
        lstm_layers:    number of LSTM layers
        attention_heads: number of attention heads
        dropout:        dropout probability
        fast_dev:       if True, runs 1 epoch on tiny data (CI / smoke test)
        register_model: if True, registers best model in MLflow registry

    Returns:
        dict with best_val_mae, best_epoch, run_id
    """
    # ── Load data ──────────────────────────────────────────────────────────
    from datetime import UTC, datetime, timedelta

    from data.ingest import build_feature_dataset

    if fast_dev:
        epochs = 1
        batch_size = 4

    end = datetime.now(UTC).replace(tzinfo=None)
    start = end - timedelta(days=60 if fast_dev else 365)
    logger.info("Loading feature dataset", country=country, start=str(start.date()))

    df = build_feature_dataset(country=country, start=start, end=end)

    # Encode country as integer
    country_map = {"DE": 0, "FR": 1, "ES": 2, "NL": 3, "PL": 4}
    df["country_id"] = country_map.get(country, 0)

    enc_len = 48 if fast_dev else settings.forecast_lookback
    dec_len = 12 if fast_dev else settings.forecast_horizon

    dataset = GridSequenceDataset(df, encoder_length=enc_len, decoder_length=dec_len)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    # Cache normalisation stats before wrapping in Subset
    numeric_mean = dataset.numeric_mean
    numeric_std  = dataset.numeric_std

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info("Dataset ready", train=n_train, val=n_val)

    # ── Model ──────────────────────────────────────────────────────────────
    config = TFTConfig(
        num_numeric_features=len(NUMERIC_COLS),
        num_categorical_features=len(CATEGORICAL_COLS),
        categorical_vocab_sizes=[6, 2],
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        attention_heads=attention_heads,
        dropout=dropout,
        encoder_length=enc_len,
        decoder_length=dec_len,
        quantiles=[0.1, 0.5, 0.9],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalFusionTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model ready", device=str(device), params=n_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )

    # ── MLflow run ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_forecast)

    best_val_mae = float("inf")
    best_epoch = 0
    best_ckpt = ARTIFACT_DIR / f"tft_best_{country}.pt"
    run_id = ""

    with mlflow.start_run(run_name=f"tft_{country}_{int(time.time())}") as run:
        run_id = run.info.run_id

        # Log hyperparameters
        mlflow.log_params({
            "country": country,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "val_split": val_split,
            **asdict(config),
        })

        for epoch in range(1, epochs + 1):
            # ── Train ──────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                enc_num = batch["enc_numeric"].to(device)
                enc_cat = batch["enc_categorical"].to(device)
                dec_num = batch["dec_numeric"].to(device)
                dec_cat = batch["dec_categorical"].to(device)
                tgt     = batch["target"].to(device)

                optimizer.zero_grad()
                out = model(enc_num, enc_cat, dec_num, dec_cat)
                loss = quantile_loss(out["quantile_forecasts"], tgt, config.quantiles)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            avg_train = train_loss / len(train_loader)

            # ── Validate ───────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            all_p50: list[torch.Tensor] = []
            all_tgt: list[torch.Tensor] = []

            with torch.no_grad():
                for batch in val_loader:
                    enc_num = batch["enc_numeric"].to(device)
                    enc_cat = batch["enc_categorical"].to(device)
                    dec_num = batch["dec_numeric"].to(device)
                    dec_cat = batch["dec_categorical"].to(device)
                    tgt     = batch["target"].to(device)

                    out = model(enc_num, enc_cat, dec_num, dec_cat)
                    loss = quantile_loss(out["quantile_forecasts"], tgt, config.quantiles)
                    val_loss += loss.item()
                    all_p50.append(out["quantile_forecasts"][:, :, 1].cpu())
                    all_tgt.append(tgt.cpu())

            avg_val = val_loss / max(len(val_loader), 1)
            scheduler.step(avg_val)

            p50_cat = torch.cat(all_p50)
            tgt_cat = torch.cat(all_tgt)

            # Denormalise for interpretable metrics
            load_std = float(numeric_std["load_mw"])
            load_mean = float(numeric_mean["load_mw"])
            p50_mw = p50_cat * load_std + load_mean
            tgt_mw = tgt_cat * load_std + load_mean

            metrics = compute_metrics(p50_mw, tgt_mw)

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": avg_train,
                "val_loss": avg_val,
                **metrics,
            }, step=epoch)

            logger.info(
                "Epoch",
                epoch=epoch,
                train_loss=round(avg_train, 4),
                val_loss=round(avg_val, 4),
                mae=round(metrics["mae"], 1),
                rmse=round(metrics["rmse"], 1),
                mape=round(metrics["mape"], 2),
            )

            # Save best checkpoint
            if metrics["mae"] < best_val_mae:
                best_val_mae = metrics["mae"]
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "metrics": metrics,
                    "numeric_mean": numeric_mean.to_dict(),
                    "numeric_std": numeric_std.to_dict(),
                }, best_ckpt)
                logger.info("New best checkpoint", mae=round(best_val_mae, 1), epoch=epoch)

        # Log best metrics and artifact
        mlflow.log_metrics({
            "best_val_mae": best_val_mae,
            "best_epoch": float(best_epoch),
        })
        mlflow.log_artifact(str(best_ckpt))

        if register_model:
            mlflow.pytorch.log_model(model, artifact_path="model")
            mv = mlflow.register_model(
                f"runs:/{run_id}/model", "voltiq_tft"
            )
            logger.info("Model registered", version=mv.version)

    logger.info(
        "Training complete",
        best_val_mae=round(best_val_mae, 1),
        best_epoch=best_epoch,
        run_id=run_id,
    )
    return {"best_val_mae": best_val_mae, "best_epoch": best_epoch, "run_id": run_id}


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Voltiq TFT forecasting model")
    parser.add_argument("--country", default="DE", choices=["DE", "FR", "ES", "NL", "PL"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fast-dev", action="store_true", help="Smoke test: 1 epoch, tiny data")
    parser.add_argument("--register", action="store_true", help="Register best model in MLflow")
    args = parser.parse_args()

    train(
        country=args.country,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        fast_dev=args.fast_dev,
        register_model=args.register,
    )