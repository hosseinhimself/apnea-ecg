## model.py

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding as described in
    "Attention Is All You Need" (Vaswani et al., 2017).
    
    Args:
        d_model (int): embedding dimension
        max_len (int): maximum length of the input sequence
        device (torch.device or str, optional): device to place positional encoding tensor
    """

    def __init__(self, d_model: int, max_len: int = 20000, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
                             (-torch.log(torch.tensor(10000.0)) / d_model))  # (d_model//2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        # Register as buffer (non-trainable)
        self.register_buffer('pe', pe)  # shape (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor of same shape as x with positional encodings added
        """
        seq_len = x.size(0)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum positional encoding length {self.max_len}."
            )
        x = x + self.pe[:seq_len, :].unsqueeze(1)  # (seq_len, 1, d_model)
        return x


class Model(nn.Module):
    """
    CNN-Transformer hybrid model for segment-wise obstructive sleep apnea detection from
    single-channel ECG signals.

    Args:
        params (dict): model hyperparameters and configuration, expected keys:
            - model.cnn.num_layers (int): Number of 1D convolutional layers (default: 10)
            - model.cnn.dropout (float): Dropout after each CNN layer and before transformer (default: 0.3)
            - model.cnn.activation (str): CNN activation function name ('relu') (default: 'relu')
            - model.transformer.dropout (float): Dropout used inside Transformer (default: 0.3)
            - model.transformer.encoder_decoder (bool): Whether to use encoder-decoder transformer (default: True)
            - model.transformer.num_encoder_layers (int): Number of encoder layers (default: 2)
            - model.transformer.num_decoder_layers (int): Number of decoder layers (default: 2)
            - model.transformer.nhead (int): Number of attention heads (default: 4)
            - model.transformer.dim_feedforward (int): Feedforward layer dimension in transformer (default: 256)
            - model.transformer.model_dim (int): Embedding dimension inside transformer (default: 64)
    """

    def __init__(self, params: Dict) -> None:
        super().__init__()

        # Extract parameters with defaults
        cnn_cfg = params.get("model", {}).get("cnn", {})
        transformer_cfg = params.get("model", {}).get("transformer", {})

        self.cnn_num_layers: int = int(cnn_cfg.get("num_layers", 10))
        self.cnn_dropout_rate: float = float(cnn_cfg.get("dropout", 0.3))
        cnn_activation_str: str = cnn_cfg.get("activation", "relu").lower()

        self.transformer_dropout_rate: float = float(transformer_cfg.get("dropout", 0.3))
        self.encoder_decoder: bool = bool(transformer_cfg.get("encoder_decoder", True))
        self.num_encoder_layers: int = int(transformer_cfg.get("num_encoder_layers", 2))
        self.num_decoder_layers: int = int(transformer_cfg.get("num_decoder_layers", 2))
        self.nhead: int = int(transformer_cfg.get("nhead", 4))
        self.dim_feedforward: int = int(transformer_cfg.get("dim_feedforward", 256))
        self.model_dim: int = int(transformer_cfg.get("model_dim", 64))  # embedding dim for transformer and CNN output channels

        # Activation function mapping
        if cnn_activation_str == "relu":
            self.cnn_activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function '{cnn_activation_str}' for CNN.")

        # Build CNN feature extractor --------------------------------------------
        # Input channels = 1 (single channel ECG)
        # Output channels for each conv layer = model_dim (64 by default)
        cnn_channels_in = 1
        cnn_channels_out = self.model_dim
        kernel_size = 3
        padding = (kernel_size - 1) // 2  # to maintain sequence length ("same" padding)

        # Using a nn.ModuleList to hold the sequence of conv layers, activations, and dropout
        cnn_layers = []
        for i in range(self.cnn_num_layers):
            conv = nn.Conv1d(
                in_channels=cnn_channels_in,
                out_channels=cnn_channels_out,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=True,
            )
            cnn_layers.append(conv)
            cnn_layers.append(self.cnn_activation)
            cnn_layers.append(nn.Dropout(p=self.cnn_dropout_rate))
            # After first layer, input channels = output channels = model_dim
            cnn_channels_in = cnn_channels_out
        self.cnn = nn.Sequential(*cnn_layers)  # processes input of shape (N, C, L)

        # Positional Encoding -----------------------------------------------------
        # Prepare positional encoding for max length large enough (e.g., 20000)
        self.positional_encoding = PositionalEncoding(
            d_model=self.model_dim, max_len=20000, device=None  # device set dynamically in forward
        )

        # Transformer Encoder-Decoder ------------------------------------------------
        # PyTorch nn.Transformer expects input shape (seq_len, batch_size, d_model)

        # Initialize Transformer model with specified configs
        # Each encoder and decoder layer uses dropout=self.transformer_dropout_rate
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout_rate,
            activation='relu',
            batch_first=False  # seq_len first for nn.Transformer
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        )

        if self.encoder_decoder:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.model_dim,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.transformer_dropout_rate,
                activation='relu',
                batch_first=False
            )
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=self.num_decoder_layers
            )
            # Learnable decoder input token (query) of shape (1, 1, model_dim) initialized randomly
            # It will be repeated across batch dimension during forward
            self.decoder_input = nn.Parameter(torch.randn(1, 1, self.model_dim))
        else:
            self.transformer_decoder = None
            self.decoder_input = None

        # Classification head -----------------------------------------------------
        # Final linear layer maps pooled transformer output to output logits
        self.classifier_dropout = nn.Dropout(p=self.cnn_dropout_rate)
        self.classifier = nn.Linear(self.model_dim, 2)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Properly initialize weights if needed: PyTorch default is usually fine
        # But can initialize conv layers with Kaiming uniform
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize decoder input to normal (already random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input ECG segments of shape (batch_size, segment_length, 1)

        Returns:
            torch.Tensor: logits of shape (batch_size, 2)
        """
        batch_size, seq_len, channels = x.shape
        if channels != 1:
            raise ValueError(f"Expected 1 channel input, got {channels} channels.")

        # Permute to (batch_size, channels=1, seq_len) for Conv1d
        x = x.permute(0, 2, 1)  # shape: (N, 1, L)

        # CNN feature extraction
        features = self.cnn(x)  # (N, model_dim, L)

        # Permute to (seq_len, batch_size, model_dim) for transformer input
        features = features.permute(2, 0, 1)  # (L, N, model_dim)

        # Move positional encoding to same device as features
        device = features.device
        if self.positional_encoding.pe.device != device:
            self.positional_encoding.pe = self.positional_encoding.pe.to(device)

        # Add positional encoding
        features = self.positional_encoding(features)  # (L, N, model_dim)

        # Transformer encoder output
        memory = self.transformer_encoder(features)  # (L, N, model_dim)

        if self.encoder_decoder:
            # Prepare decoder input: learned parameter repeated along batch dimension
            # decoder_input shape: (1, batch_size, model_dim)
            tgt = self.decoder_input.repeat(1, batch_size, 1)  # (1, N, model_dim)
            # Generate output from transformer decoder
            output = self.transformer_decoder(tgt=tgt, memory=memory)  # (1, N, model_dim)
            # Remove sequence dimension (which is 1), permute to (N, model_dim)
            output = output.squeeze(0).permute(0, 1)  # (N, model_dim)
        else:
            # If no decoder, use encoder output: pool over sequence (dim=0)
            # shape encoder output: (L, N, model_dim)
            output = memory.mean(dim=0)  # (N, model_dim)

        # Classification head
        output = self.classifier_dropout(output)  # (N, model_dim)
        logits = self.classifier(output)  # (N, 2)

        return logits
