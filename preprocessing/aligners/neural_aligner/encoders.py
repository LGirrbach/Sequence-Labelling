"""
Stores different sequences encoders:
Each encoder encodes a sequence of embeddings. Currently, 2 implementations are available:
 * Bidirectional LSTM
 * Convolutional network
"""

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = None

    @staticmethod
    def encoder_factory(encoder_type, input_size, **parameters):
        if encoder_type == 'conv':
            return ConvEncoder(input_size, **parameters)
        elif encoder_type == 'lstm':
            return LSTMEncoder(input_size, **parameters)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")


class LSTMEncoder(Encoder):
    """
    Wrapper around LSTM that detects whether input is a batch of sequences or a single sequence and handles
    packing/unpacking of padded sequences.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0, **params):
        super().__init__()
        self.params = params
        # Bidirectional LSTM doubles hidden size
        self.output_size = 2 * hidden_size

        # Maintain trainable parameters for first LSTM hidden/cell states
        self.h_0 = nn.Parameter(torch.zeros(2 * num_layers, 1, hidden_size))
        self.c_0 = nn.Parameter(torch.zeros(2 * num_layers, 1, hidden_size))

        # Instantiate LSTM
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, inputs, lengths=None):
        # Check if input is batch or not
        num_dimensions = len(inputs.shape)

        if num_dimensions == 3:
            # Assume batch first
            batch_size = inputs.shape[0]
            # Pack sequence, if lengths are given
            if lengths is not None:
                lengths = torch.clamp(lengths, 1)
                inputs = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)

        elif num_dimensions == 2:
            # Add dummy batch dimension
            batch_size = 1
            inputs = inputs.unsqueeze(0)

        else:
            raise RuntimeError(f"Input to LSTM must have either 2 or 3 dims, but has {num_dimensions}")

        # Apply LSTM
        h_0 = self.h_0.tile((1, batch_size, 1))
        c_0 = self.c_0.tile((1, batch_size, 1))
        encoded, _ = self.lstm(inputs, (h_0, c_0))

        if num_dimensions == 3 and lengths is not None:
            # Unpack packed sequence
            encoded, _ = pad_packed_sequence(encoded, batch_first=True)

        elif num_dimensions == 2:
            # Remove dummy dimension
            encoded = encoded.squeeze(0)

        return encoded


class ConvEncoder(Encoder):
    """
    Multiple layers of 1d-convolutions.
    Sequences are 0-padded to leave lengths unchanged by convolutions.
    Each convolution block consists of the following 4 operations:
      1. Batch norm
      2. Dropout (can be disabled by setting dropout = 0.0)
      3. 1d-Convolution
      4. ReLU nonlinearity

    Usually, BatchNorm is applied either before or after activation. Similar to dropout, I have placed BatchNorm
    at the very start of the convolution block, in order to avoid dropout or BatchNorm being the last operations
    when encoding the sequence.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0,
                 kernel_size: int = 3, batch_norm: bool = False, **params):
        super().__init__()
        self.params = params
        self.output_size = hidden_size

        modules = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            kernel = kernel_size if layer == 0 else 1

            if batch_norm:
                modules.append(nn.BatchNorm1d(num_features=in_size))

            modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Conv1d(in_channels=in_size, out_channels=hidden_size, kernel_size=(kernel,),
                                     padding='same'))
            modules.append(nn.ReLU())

        self.conv_net = nn.Sequential(*modules)

    def forward(self, inputs, lengths=None):
        # Check if input is batch or not
        num_dimensions = len(inputs.shape)

        if num_dimensions == 3:
            # Assume batch dim first, swap feature and timesteps dims
            inputs = inputs.transpose(1, 2)

        elif num_dimensions == 2:
            # Add dummy batch dimension
            inputs = inputs.unsqueeze(0)
            # Assume batch dim first, swap feature and timesteps dims
            inputs = inputs.transpose(1, 2)

        else:
            raise RuntimeError(f"Input to LSTM must have either 2 or 3 dims, but has {num_dimensions}")

        encoded = self.conv_net(inputs)
        # Restore batch, timesteps, features
        encoded = encoded.transpose(1, 2)

        # Remove dummy batch dimension
        if num_dimensions == 2:
            encoded = encoded.squeeze(0)

        return encoded
