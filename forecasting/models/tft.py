"""
forecasting/models/tft.py — Temporal Fusion Transformer for grid load forecasting.

Simplified TFT implementation in pure PyTorch, designed to:
  - Run inference on CPU (no GPU required locally)
  - Train on Colab T4 in ~1-2 hours on a year of hourly data
  - Produce calibrated 48-hour ahead point + interval forecasts

Architecture:
  Input embeddings (numeric + categorical)
  → Variable Selection Networks (VSN)
  → LSTM encoder-decoder
  → Multi-head Attention (temporal self-attention)
  → Gated Residual Networks
  → Quantile output heads (p10, p50, p90)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


@dataclass
class TFTConfig:
    num_numeric_features: int = 7
    num_categorical_features: int = 2
    categorical_vocab_sizes: list[int] | None = None

    hidden_size: int = 64
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1

    encoder_length: int = 168
    decoder_length: int = 48

    quantiles: list[float] | None = None

    def __post_init__(self) -> None:
        if self.categorical_vocab_sizes is None:
            self.categorical_vocab_sizes = [6, 2]
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]


class GatedResidualNetwork(nn.Module):
    """GRN: core building block of TFT. Applies gating to skip trivial transformations."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        out = gate * out
        return self.layer_norm(out + residual)


class VariableSelectionNetwork(nn.Module):
    """VSN: learns which input variables matter most at each time step."""

    def __init__(
        self, input_dim: int, num_inputs: int, hidden_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        self.single_variable_grns = nn.ModuleList(
            [
                GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
                for _ in range(num_inputs)
            ]
        )
        self.variable_selector = GatedResidualNetwork(
            input_dim * num_inputs, hidden_dim, num_inputs, dropout
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, num_inputs * input_dim)
        Returns: (processed, variable_weights)
        """
        flat = x.reshape(x.size(0), x.size(1), -1)
        weights = F.softmax(self.variable_selector(flat), dim=-1)

        processed = torch.stack(
            [
                grn(x[..., i : i + 1].expand(-1, -1, x.size(-1)))
                for i, grn in enumerate(self.single_variable_grns)
            ],
            dim=-1,
        )

        out = (processed * weights.unsqueeze(-2)).sum(-1)
        return out, weights


class TemporalSelfAttention(nn.Module):
    """Interpretable multi-head attention — single head per query for interpretability."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, dim = x.shape
        n_heads, head_d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(batch, seq, n_heads, head_d).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, n_heads, head_d).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, n_heads, head_d).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, seq, dim)
        return self.out_proj(out), attn.mean(dim=1)


class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for grid load forecasting.

    Forward pass returns:
      - quantile_forecasts: (batch, decoder_len, num_quantiles)
      - attention_weights:  (batch, decoder_len, encoder_len) — interpretable!
      - variable_weights:   (batch, seq_len, num_inputs) — feature importance
    """

    def __init__(self, config: TFTConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, hidden) for vocab_size in config.categorical_vocab_sizes]
        )

        self.numeric_proj = nn.Linear(config.num_numeric_features, hidden)

        total_features = config.num_numeric_features + config.num_categorical_features

        self.encoder_vsn = VariableSelectionNetwork(hidden, total_features, hidden, config.dropout)
        self.decoder_vsn = VariableSelectionNetwork(hidden, total_features, hidden, config.dropout)

        self.encoder_lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.attention = TemporalSelfAttention(hidden, config.attention_heads, config.dropout)
        self.attn_norm = nn.LayerNorm(hidden)
        self.pos_grn = GatedResidualNetwork(hidden, hidden * 2, hidden, config.dropout)

        self.output_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in config.quantiles])

    def _embed_inputs(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        """Embed and concatenate all input features."""
        num_emb = self.numeric_proj(numeric)
        cat_embs = [emb(categorical[..., i]) for i, emb in enumerate(self.cat_embeddings)]
        all_features = torch.stack([num_emb] + cat_embs, dim=-1)
        return all_features.permute(0, 1, 3, 2).reshape(*all_features.shape[:2], -1)

    def forward(
        self,
        enc_numeric: torch.Tensor,
        enc_categorical: torch.Tensor,
        dec_numeric: torch.Tensor,
        dec_categorical: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        enc_in = self._embed_inputs(enc_numeric, enc_categorical)
        dec_in = self._embed_inputs(dec_numeric, dec_categorical)

        enc_selected, enc_var_wt = self.encoder_vsn(enc_in)
        dec_selected, dec_var_wt = self.decoder_vsn(dec_in)

        enc_out, hidden_state = self.encoder_lstm(enc_selected)
        dec_out, _ = self.decoder_lstm(dec_selected, hidden_state)

        full_seq = torch.cat([enc_out, dec_out], dim=1)
        t_enc = enc_out.size(1)
        t_total = full_seq.size(1)

        mask = torch.triu(torch.ones(t_total, t_total, dtype=torch.bool), diagonal=1)
        mask = mask.to(enc_out.device)

        attn_out, attn_weights = self.attention(full_seq, mask)
        attn_out = self.attn_norm(attn_out + full_seq)

        dec_attn = attn_out[:, t_enc:, :]
        dec_final = self.pos_grn(dec_attn)

        quantile_preds = torch.cat([head(dec_final) for head in self.output_heads], dim=-1)

        return {
            "quantile_forecasts": quantile_preds,
            "attention_weights": attn_weights[:, t_enc:, :t_enc],
            "encoder_variable_weights": enc_var_wt,
            "decoder_variable_weights": dec_var_wt,
        }
