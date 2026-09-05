"""
anomaly/detector.py — LSTM Autoencoder for smart-meter anomaly detection.

Unsupervised: train on normal data, flag high reconstruction-error windows.
CPU-friendly. Colab T4 recommended for large datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

from core.logging import get_logger

logger = get_logger(__name__)


class TimeSeriesWindowDataset(Dataset):
    """Sliding window dataset over a 1D time series."""

    def __init__(self, series: np.ndarray, window_size: int = 24) -> None:
        self.windows = [
            torch.tensor(series[i : i + window_size], dtype=torch.float32).unsqueeze(-1)
            for i in range(len(series) - window_size + 1)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, latent_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out, (h, c) = self.lstm(x)
        latent = self.fc(out[:, -1, :])
        return latent, (h, c)


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        h = self.fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(h)
        return self.out_proj(out)


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder — high reconstruction error flags anomalies."""

    def __init__(
        self,
        window_size: int = 24,
        input_size: int = 1,
        hidden_size: int = 64,
        latent_dim: int = 16,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, num_layers, input_size, window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.encoder(x)
        return self.decoder(latent)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE — returns CPU float tensor, no numpy needed."""
        recon = self(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2])


class AnomalyDetector:
    """High-level wrapper: train, calibrate threshold, detect."""

    def __init__(
        self,
        window_size: int = 24,
        threshold_percentile: float = 95.0,
        model_path: Path | None = None,
    ) -> None:
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.threshold: float | None = None
        self.model = LSTMAutoencoder(window_size=window_size)
        self.device = torch.device("cpu")

        if model_path and model_path.exists():
            self.load(model_path)

    def train(
        self,
        normal_series: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> list[float]:
        """Train on normal data. Returns per-epoch losses."""
        dataset = TimeSeriesWindowDataset(normal_series, self.window_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.to(self.device).train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = F.mse_loss(recon, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(loader)
            losses.append(avg)
            scheduler.step(avg)
            if (epoch + 1) % 5 == 0:
                logger.info("Anomaly training", epoch=epoch + 1, loss=round(avg, 6))

        self._calibrate_threshold(normal_series)
        return losses

    def _calibrate_threshold(self, series: np.ndarray) -> None:
        dataset = TimeSeriesWindowDataset(series, self.window_size)
        loader = DataLoader(dataset, batch_size=256)
        errors: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                err = self.model.reconstruction_error(batch.to(self.device))
                # Use .tolist() directly — avoids numpy/torch ABI incompatibility
                errors.extend(err.tolist())
        self.threshold = float(np.percentile(errors, self.threshold_percentile))
        logger.info(
            "Anomaly threshold calibrated",
            threshold=round(self.threshold, 6),
            percentile=self.threshold_percentile,
        )

    def detect(self, series: np.ndarray) -> dict[str, Any]:
        """Detect anomalies. Returns indices, errors, threshold, rate."""
        if self.threshold is None:
            raise RuntimeError("Call train() first or load a saved model.")

        dataset = TimeSeriesWindowDataset(series, self.window_size)
        loader = DataLoader(dataset, batch_size=256)
        errors: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                err = self.model.reconstruction_error(batch.to(self.device))
                errors.extend(err.tolist())

        errors_arr = np.array(errors)
        anomaly_indices = [int(i) for i in np.where(errors_arr > self.threshold)[0]]
        return {
            "anomaly_indices": anomaly_indices,
            "reconstruction_errors": errors_arr.tolist(),
            "threshold": self.threshold,
            "anomaly_rate": len(anomaly_indices) / max(len(errors_arr), 1),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "window_size": self.window_size,
                    "threshold": self.threshold,
                    "threshold_percentile": self.threshold_percentile,
                }
            )
        )
        logger.info("Anomaly model saved", path=str(path))

    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        meta = path.with_suffix(".meta.json")
        if meta.exists():
            data = json.loads(meta.read_text())
            self.threshold = data.get("threshold")
            self.window_size = data.get("window_size", self.window_size)
        logger.info("Anomaly model loaded", path=str(path), threshold=self.threshold)
