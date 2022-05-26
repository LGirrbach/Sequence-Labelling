import torch

from typing import Any
from blocks import TransformerBlock
from encoders.encoder import Encoder
from blocks import PositionalEncoding


class TransformerEncoder(Encoder):
    """Implements a (non-conditional) transformer encoder for sequence labelling."""
    def __init__(self, embedding_size: int = 32, hidden_size: int = 128, num_layers: int = 1,
                 dim_feedforward: int = 2048, nhead: int = 4, dropout: float = 0.0):
        super(TransformerEncoder, self).__init__()

        # Save arguments
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.dropout = dropout

    def get_save_info(self):
        return {
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'nhead': self.nhead,
            'dropout': self.dropout,
        }

    def _initialise_embedding_layer(self, vocab_size: int):
        super(TransformerEncoder, self)._initialise_embedding_layer(vocab_size)
        self.positional_encodings = PositionalEncoding(self.embedding_size, self.dropout, 200)

    def _initialise_encoder(self):
        # Encoder only consists of one transformer block
        self.transformer = TransformerBlock(
            self.embedding_size, self.hidden_size, self.num_layers, self.dim_feedforward, self.nhead, self.dropout
        )

    def _embed(self, inputs: torch.Tensor):
        embedded = super(TransformerEncoder, self)._embed(inputs)
        embedded = self.positional_encodings(embedded)
        return embedded

    def _apply_encoder(self, inputs: torch.Tensor, lengths: torch.Tensor, conditions: Any = None):
        return self.transformer(inputs, lengths)
