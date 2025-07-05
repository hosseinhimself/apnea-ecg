## model.py

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20000, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
                             (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum positional encoding length {self.max_len}."
            )
        return x + self.pe[:seq_len, :].unsqueeze(1).to(x.dtype)


class Model(nn.Module):
    def __init__(self, params: Dict) -> None:
        super().__init__()

        cnn_cfg = params.get("model", {}).get("cnn", {})
        transformer_cfg = params.get("model", {}).get("transformer", {})

        self.cnn_num_layers: int = int(cnn_cfg.get("num_layers", 10))
        self.cnn_dropout_rate: float = float(cnn_cfg.get("dropout", 0.1))
        cnn_activation_str: str = cnn_cfg.get("activation", "relu").lower()

        self.transformer_dropout_rate: float = float(transformer_cfg.get("dropout", 0.3))
        self.num_encoder_layers: int = int(transformer_cfg.get("num_encoder_layers", 2))
        self.nhead: int = int(transformer_cfg.get("nhead", 2))
        self.dim_feedforward: int = int(transformer_cfg.get("dim_feedforward", 128))
        self.model_dim: int = int(transformer_cfg.get("model_dim", 32))

        if cnn_activation_str == "relu":
            self.cnn_activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function '{cnn_activation_str}' for CNN.")

        cnn_channels_in = 1
        cnn_channels_out = self.model_dim
        cnn_layers = []

        for i in range(self.cnn_num_layers):
            layers = [
                nn.Conv1d(cnn_channels_in, cnn_channels_out, kernel_size=32, padding=16),
                nn.BatchNorm1d(cnn_channels_out),
                nn.ReLU(inplace=True)
            ]
            if i < 6:
                layers.append(nn.MaxPool1d(kernel_size=2))
            layers.append(nn.Dropout(p=self.cnn_dropout_rate))
            cnn_layers.extend(layers)
            cnn_channels_in = cnn_channels_out

        self.cnn = nn.Sequential(*cnn_layers)

        self.positional_encoding = PositionalEncoding(
            d_model=self.model_dim, max_len=20000, device=None
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout_rate,
            activation='relu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        )

        self.classifier_dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.model_dim, 2)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        if channels != 1:
            raise ValueError(f"Expected 1 channel input, got {channels} channels.")

        x = x.permute(0, 2, 1)  # (N, 1, L)
        features = self.cnn(x)  # (N, model_dim, L)
        features = features.permute(2, 0, 1)  # (L, N, model_dim)

        device = features.device
        if self.positional_encoding.pe.device != device:
            self.positional_encoding.pe = self.positional_encoding.pe.to(device)

        features = self.positional_encoding(features)
        memory = self.transformer_encoder(features)
        output = memory.mean(dim=0)  # Global average pooling
        output = self.classifier_dropout(output)

        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)  # safety net
        logits = self.classifier(output)
        return logits
