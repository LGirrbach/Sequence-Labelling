import torch

from typing import Any
from blocks import ConvBlock
from encoders.encoder import Encoder


class ConvEncoder(Encoder):
    """Implements a (non-conditional) convolutional encoder for sequence labelling"""
    def __init__(self, embedding_size: int = 32, hidden_size: int = 128, num_layers: int = 1, kernel_size: int = 3,
                 dropout: float = 0.0, batch_norm: bool = False, residual: bool = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

    def get_save_info(self):
        return {
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'residual': self.residual
        }

    def _initialise_encoder(self):
        # Encoder only consists of one convolutional block
        self.conv = ConvBlock(
            self.embedding_size, self.hidden_size, self.num_layers, self.kernel_size, self.dropout, self.batch_norm,
            self.residual
        )

    def _apply_encoder(self, inputs: torch.Tensor, lengths: torch.Tensor, conditions: Any = None):
        return self.conv(inputs)
