import torch

from typing import Any
from typing import List
from typing import Union
from encoders import Encoder
from blocks import BiLSTMBlock
from blocks import MultiConvBlock


class NgramLSTMEncoder(Encoder):
    """Implements a (non-conditional) ngram encoder followed by a LSTM encoder for sequence labelling"""
    def __init__(self, embedding_size: int = 32, hidden_size: int = 128, num_layers: int = 1,
                 kernel_sizes: Union[int, List[int]] = 3,  dropout: float = 0.0, batch_norm: bool = False,
                 cond_dim: int = None):
        super(NgramLSTMEncoder, self).__init__()

        # Save arguments
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.cond_dim = cond_dim

    def get_save_info(self):
        return {
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'kernel_sizes': self.kernel_sizes,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'cond_dim': self.cond_dim,
        }

    def _initialise_encoder(self):
        # Make ngram feature encoders (1d-convolutions)
        self.ngram_encoder = MultiConvBlock(
            self.embedding_size, self.hidden_size, self.kernel_sizes, self.dropout, self.batch_norm
        )
        # Make lstm
        self.lstm = BiLSTMBlock(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout
        )

    def _apply_encoder(self, inputs: torch.Tensor, lengths: torch.Tensor, conditions: Any = None):
        # Apply convolutional ngram encoder
        encoded = self.ngram_encoder(inputs)
        # Prepare condition as initial hidden state (optional)
        hidden = None if self.cond_dim is None else conditions
        # Apply LSTM
        encoded = self.lstm(encoded, lengths, hidden)

        return encoded
