"""
forecasting/models/tft.py — Temporal Fusion Transformer for grid load forecasting.

CPU-friendly: runs inference on CPU, train on Colab T4.
Produces 48h-ahead point + interval forecasts (p10, p50, p90).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


@dataclass
class TFTConfig:
    num_numeric_features: int = 7
    num_categorical_features: int = 2
    categorical_vocab_sizes: list[int] = field(default_factory=lambda: [6, 2])
    hidden_size: int = 64
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1
    encoder_length: int = 168
    decoder_length: int = 48
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


class GatedResidualNetwork(nn.Module):
    """GRN: core building block of TFT."""

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
        return self.layer_norm(gate * out + residual)


class VariableSelectionNetwork(nn.Module):
    """VSN: learns which input variables matter most at each time step."""

    def __init__(
        self, hidden_dim: int, num_inputs: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        # Each variable gets its own GRN(hidden_dim -> hidden_dim)
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_inputs)
        ])
        # Selector over concatenated variables
        self.selector = GatedResidualNetwork(
            hidden_dim * num_inputs, hidden_dim, num_inputs, dropout
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, num_inputs, hidden_dim)
        Returns: processed (batch, seq_len, hidden_dim), weights (batch, seq_len, num_inputs)
        """
        # x shape: (B, T, num_inputs, hidden_dim)
        batch, seq, n_inp, h_dim = x.shape

        # Flatten inputs for selector: (B, T, num_inputs * hidden_dim)
        flat = x.reshape(batch, seq, n_inp * h_dim)
        weights = F.softmax(self.selector(flat), dim=-1)  # (B, T, num_inputs)

        # Process each variable independently
        var_outputs = torch.stack([
            self.var_grns[i](x[:, :, i, :])  # (B, T, hidden_dim)
            for i in range(self.num_inputs)
        ], dim=2)  # (B, T, num_inputs, hidden_dim)

        # Weighted sum: (B, T, hidden_dim)
        out = (var_outputs * weights.unsqueeze(-1)).sum(dim=2)
        return out, weights


class TemporalSelfAttention(nn.Module):
    """Interpretable multi-head self-attention."""

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
        n_h, h_d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(batch, seq, n_h, h_d).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, n_h, h_d).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, n_h, h_d).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, seq, dim)
        return self.out_proj(out), attn.mean(dim=1)


class TemporalFusionTransformer(nn.Module):
    """
    TFT for grid load forecasting.
    Returns quantile_forecasts, attention_weights, variable_weights.
    """

    def __init__(self, config: TFTConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        total = config.num_numeric_features + config.num_categorical_features

        # Project numeric features → hidden_dim each
        self.numeric_proj = nn.Linear(1, hidden)

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab, hidden)
            for vocab in config.categorical_vocab_sizes
        ])

        # Variable selection (operates on (B, T, num_inputs, hidden))
        self.encoder_vsn = VariableSelectionNetwork(hidden, total, config.dropout)
        self.decoder_vsn = VariableSelectionNetwork(hidden, total, config.dropout)

        # LSTM encoder-decoder
        self.encoder_lstm = nn.LSTM(
            hidden, hidden, config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder_lstm = nn.LSTM(
            hidden, hidden, config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.attention = TemporalSelfAttention(hidden, config.attention_heads, config.dropout)
        self.attn_norm = nn.LayerNorm(hidden)
        self.pos_grn = GatedResidualNetwork(hidden, hidden * 2, hidden, config.dropout)

        self.output_heads = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in config.quantiles
        ])

    def _embed(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        """
        numeric:     (B, T, num_numeric)
        categorical: (B, T, num_categorical)
        Returns:     (B, T, total_features, hidden)
        """
        # Each numeric feature → (B, T, hidden) via shared linear on (B, T, 1)
        num_embs = [
            self.numeric_proj(numeric[:, :, i].unsqueeze(-1))
            for i in range(numeric.shape[-1])
        ]
        cat_embs = [
            self.cat_embeddings[i](categorical[:, :, i])
            for i in range(categorical.shape[-1])
        ]
        all_embs = num_embs + cat_embs  # list of (B, T, hidden)
        return torch.stack(all_embs, dim=2)  # (B, T, total, hidden)

    def forward(
        self,
        enc_numeric: torch.Tensor,
        enc_categorical: torch.Tensor,
        dec_numeric: torch.Tensor,
        dec_categorical: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        enc_in = self._embed(enc_numeric, enc_categorical)  # (B, T_enc, total, hidden)
        dec_in = self._embed(dec_numeric, dec_categorical)  # (B, T_dec, total, hidden)

        enc_sel, enc_var_wt = self.encoder_vsn(enc_in)   # (B, T_enc, hidden)
        dec_sel, dec_var_wt = self.decoder_vsn(dec_in)   # (B, T_dec, hidden)

        enc_out, hidden_state = self.encoder_lstm(enc_sel)
        dec_out, _ = self.decoder_lstm(dec_sel, hidden_state)

        full_seq = torch.cat([enc_out, dec_out], dim=1)
        t_enc = enc_out.size(1)
        t_total = full_seq.size(1)

        mask = torch.triu(torch.ones(t_total, t_total, dtype=torch.bool), diagonal=1)
        attn_out, attn_weights = self.attention(full_seq, mask.to(enc_out.device))
        attn_out = self.attn_norm(attn_out + full_seq)

        dec_final = self.pos_grn(attn_out[:, t_enc:, :])
        quantile_preds = torch.cat(
            [head(dec_final) for head in self.output_heads], dim=-1
        )

        return {
            "quantile_forecasts": quantile_preds,
            "attention_weights": attn_weights[:, t_enc:, :t_enc],
            "encoder_variable_weights": enc_var_wt,
            "decoder_variable_weights": dec_var_wt,
        }
