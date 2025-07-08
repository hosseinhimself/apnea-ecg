# model.py

from typing import Dict, Optional

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding, which adds information about the
    relative or absolute position of the tokens in the sequence.
    This is crucial for Transformer models as they do not inherently
    process sequences in order.
    """
    def __init__(self, d_model: int, max_len: int = 20000, device: Optional[torch.device] = None) -> None:
        super().__init__()
        # Create a zero tensor for positional encodings
        pe = torch.zeros(max_len, d_model, device=device)
        # Calculate positions
        position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
        # Calculate the divisor term for the sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        # Apply sine to even indices in the positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the positional encoding
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register 'pe' as a buffer, meaning it's part of the module's state
        # but not a learnable parameter.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to which positional encoding will be added.
                              Expected shape: (sequence_length, batch_size, d_model).

        Returns:
            torch.Tensor: The input tensor with positional encoding added.
        """
        seq_len = x.size(0)
        # Add positional encoding to the input. unsqueeze(1) makes it broadcastable.
        return x + self.pe[:seq_len, :].unsqueeze(1).to(x.dtype)

class Model(nn.Module):
    """
    The main CNN-Transformer model architecture for Obstructive Sleep Apnea (OSA) detection
    from single-channel ECG signals, as proposed in the paper.
    It comprises a CNN for representation learning and a Transformer Encoder for sequence modeling.
    """
    def __init__(self, params: Dict) -> None:
        """
        Initializes the CNN-Transformer model.

        Args:
            params (Dict): A dictionary containing model configuration parameters.
        """
        super().__init__()

        # Extract CNN and Transformer specific parameters from the config
        cnn_cfg = params.get("model", {}).get("cnn", {})
        transformer_cfg = params.get("model", {}).get("transformer", {})

        # CNN parameters as specified in Figure 2 and Section 2.2 of the paper
        self.cnn_dropout_rate: float = float(cnn_cfg.get("dropout", 0.3))
        self.cnn_kernel_size: int = 32
        self.cnn_padding: int = 16
        
        # Transformer parameters as specified in Section 2.3 and 2.3.1 of the paper
        self.model_dim: int = 64 # This is d_model, the dimension of the model's embeddings
        self.nhead: int = 2 # Number of attention heads
        self.num_encoder_layers: int = 2 # Number of Transformer encoder layers (N in paper)
        self.dim_feedforward: int = 128 # Dimension of the feedforward network in Transformer
        self.transformer_dropout_rate: float = float(transformer_cfg.get("dropout", 0.3))

        # --- Build the CNN part EXACTLY as described in Figure 2 and Section 2.2 of the paper ---
        cnn_layers = []
        in_ch = 1 # Input channel for the first CNN layer (single-channel ECG)

        # First 4 blocks: Conv1D -> BatchNorm1d -> ReLU -> MaxPool1d -> Dropout
        for i in range(4):
            cnn_layers.extend([
                nn.Conv1d(in_ch, self.model_dim, kernel_size=self.cnn_kernel_size, padding=self.cnn_padding),
                nn.BatchNorm1d(self.model_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(p=self.cnn_dropout_rate)
            ])
            in_ch = self.model_dim # Output channels of previous layer become input for next

        # Next 6 blocks: Conv1D -> BatchNorm1d -> ReLU -> Dropout (no MaxPool1d)
        for i in range(6):
            cnn_layers.extend([
                nn.Conv1d(self.model_dim, self.model_dim, kernel_size=self.cnn_kernel_size, padding=self.cnn_padding),
                nn.BatchNorm1d(self.model_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.cnn_dropout_rate)
            ])
        # Combine all CNN layers into a sequential module
        self.cnn = nn.Sequential(*cnn_layers)
        # --- End of CNN block ---

        # Initialize Positional Encoding layer
        self.positional_encoding = PositionalEncoding(d_model=self.model_dim, max_len=20000)

        # Create a single Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout_rate,
            activation='relu', # ReLU activation for the FFN in the encoder
            batch_first=False # Input expected as (sequence_length, batch_size, features)
        )
        # Stack multiple encoder layers to form the Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)

        # Decoder part: Global Average Pooling (handled in forward) + Feed-Forward Network (FFN)
        # As described in Section 2.3.2, the FFN consists of two linear layers.
        self.decoder = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim), # First linear layer
            nn.ReLU(inplace=True), # ReLU activation
            nn.Linear(self.model_dim, 2) # Second linear layer, output dimension is 2 (normal/apnea classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CNN-Transformer model.

        Args:
            x (torch.Tensor): Input ECG signal tensor. Expected shape: (batch_size, sequence_length, 1).

        Returns:
            torch.Tensor: Logits for the two classes (normal/apnea).
        """
        # Permute input for CNN: (batch_size, channels, sequence_length)
        # Original: (batch_size, sequence_length, 1) -> Desired: (batch_size, 1, sequence_length)
        x = x.permute(0, 2, 1)
        
        # Pass through the CNN to learn feature representations
        features = self.cnn(x)
        
        # Permute features for Transformer: (sequence_length, batch_size, features_dim)
        # Original: (batch_size, features_dim, sequence_length) -> Desired: (sequence_length, batch_size, features_dim)
        features = features.permute(2, 0, 1)

        # Add positional encoding to the features
        features = self.positional_encoding(features)
        
        # Pass features through the Transformer Encoder
        memory = self.transformer_encoder(features)
        
        # Apply Global Average Pooling (GAP) as the first step of the decoder
        # This averages the features across the sequence dimension (dim=0)
        output = memory.mean(dim=0)
        
        # Pass the pooled output through the FFN decoder to get logits
        logits = self.decoder(output)
        return logits