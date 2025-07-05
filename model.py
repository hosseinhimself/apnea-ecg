## model.py

from typing import Dict, Optional

import torch
import torch.nn as nn

# PositionalEncoding class remains the same...
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20000, device: Optional[torch.device] = None) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :].unsqueeze(1).to(x.dtype)

class Model(nn.Module):
    def __init__(self, params: Dict) -> None:
        super().__init__()

        # <<< FIX: Parameters must match the paper EXACTLY >>>
        cnn_cfg = params.get("model", {}).get("cnn", {})
        transformer_cfg = params.get("model", {}).get("transformer", {})

        # CNN parameters from paper
        self.cnn_dropout_rate: float = float(cnn_cfg.get("dropout", 0.3))
        self.cnn_kernel_size: int = 32
        self.cnn_padding: int = 16
        
        # Transformer parameters from paper
        self.model_dim: int = 64 # This is d_model in the paper
        self.nhead: int = 2
        self.num_encoder_layers: int = 2
        self.dim_feedforward: int = 128
        self.transformer_dropout_rate: float = float(transformer_cfg.get("dropout", 0.3))

        # --- Build the CNN part EXACTLY as in Figure 2 of the paper ---
        cnn_layers = []
        # First 4 blocks with Conv1D, BN, ReLU, MaxPool, Dropout
        in_ch = 1
        for i in range(4):
            cnn_layers.extend([
                nn.Conv1d(in_ch, self.model_dim, kernel_size=self.cnn_kernel_size, padding=self.cnn_padding),
                nn.BatchNorm1d(self.model_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(p=self.cnn_dropout_rate)
            ])
            in_ch = self.model_dim # After the first layer, in_channels is model_dim

        # Next 6 blocks with Conv1D, BN, ReLU, Dropout (no MaxPool)
        for i in range(6):
            cnn_layers.extend([
                nn.Conv1d(self.model_dim, self.model_dim, kernel_size=self.cnn_kernel_size, padding=self.cnn_padding),
                nn.BatchNorm1d(self.model_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.cnn_dropout_rate)
            ])
        self.cnn = nn.Sequential(*cnn_layers)
        # --- End of CNN block ---

        self.positional_encoding = PositionalEncoding(d_model=self.model_dim, max_len=20000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout_rate,
            activation='relu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)

        # Decoder part as described in the paper (GAP + FFN)
        self.decoder = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim), # Paper mentions two linear layers
            nn.ReLU(inplace=True),
            nn.Linear(self.model_dim, 2) # Output dimension is 2 (apnea/normal)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        features = self.cnn(x)
        features = features.permute(2, 0, 1)

        features = self.positional_encoding(features)
        memory = self.transformer_encoder(features)
        
        # Global Average Pooling
        output = memory.mean(dim=0)
        
        logits = self.decoder(output)
        return logits
    