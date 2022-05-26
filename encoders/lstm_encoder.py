import torch

from typing import Any
from blocks import BiLSTMBlock
from encoders.encoder import Encoder


class LSTMEncoder(Encoder):
    """Implements a BiLSTM encoder for sequence labelling."""
    def __init__(self, embedding_size: int = 32, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.0,
                 cond_dim: int = None):
        super(LSTMEncoder, self).__init__()

        # Save arguments
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.cond_dim = cond_dim

    def get_save_info(self):
        return {
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'cond_dim': self.cond_dim,
        }

    def _initialise_encoder(self):
        # Encoder only consists of 1 LSTM block
        self.lstm = BiLSTMBlock(
            input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout
        )

    def _apply_encoder(self, inputs: torch.Tensor, lengths: torch.Tensor, conditions: Any = None):
        # Prepare condition as initial hidden state (optional)
        hidden = None if self.cond_dim is None else conditions
        # Apply LSTM
        encoded = self.lstm(inputs, lengths, hidden)

        return encoded
