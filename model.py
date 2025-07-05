import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=32, stride=1, padding=16):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if residual.shape == x.shape:
            x = x + residual
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        model_cfg = params.get("model", {})
        cnn_cfg = model_cfg.get("cnn", {})
        transformer_cfg = model_cfg.get("transformer", {})

        # CNN Configuration
        self.cnn_layers = nn.Sequential(
            *[CNNBlock(1 if i == 0 else 64, 64) 
              for i in range(cnn_cfg.get("num_layers", 6))]
        )

        # Transformer Configuration
        self.model_dim = transformer_cfg.get("model_dim", 64)
        self.pos_encoder = PositionalEncoding(self.model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=transformer_cfg.get("nhead", 4),
            dim_feedforward=transformer_cfg.get("dim_feedforward", 256),
            dropout=transformer_cfg.get("dropout", 0.3),
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_cfg.get("num_encoder_layers", 2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, 2)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)  # [N, 1, L]
        x = self.cnn_layers(x)
        x = x.permute(2, 0, 1)  # [L, N, D]
        
        # Transformer
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling
        
        # Classifier
        x = self.classifier(x)
        return x
