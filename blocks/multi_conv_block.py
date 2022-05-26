import torch
import torch.nn as nn

from typing import List
from typing import Union
from functools import partial
from blocks.conv_block import ConvLayer


class MultiConvBlock(nn.Module):
    """
    Implements a ngram encoder:
    Ngram features of a given sequence are calculated by applying 1d-convolutions with different kernel sizes.
    Finally, all ngram filters are concatenated and projected to the common hidden dimension.
    Note: This implementation assumes batch first.
    """
    def __init__(self, input_size: int, hidden_size: int = 128, kernel_sizes: Union[int, List[int]] = 3,
                 dropout: float = 0.0, batch_norm: bool = False):
        super(MultiConvBlock, self).__init__()

        # Save arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.batch_norm = batch_norm

        # If necessary, expand the given max. ngram size into a list of kernel sizes
        if isinstance(self.kernel_sizes, int):
            assert self.kernel_sizes >= 2
            kernel_sizes = list(range(2, self.kernel_sizes))
        # Check type correctness
        elif not isinstance(kernel_sizes, list) or not all(isinstance(size, int) for size in kernel_sizes):
            raise TypeError(f"Kernel sizes must be int or List[int], not {type(kernel_sizes)}")

        # Make modules
        make_conv_block = partial(ConvLayer, input_size=input_size, hidden_size=hidden_size, dropout=dropout,
                                  batch_norm=batch_norm)
        self.convolutions = nn.ModuleList([make_conv_block(kernel_size=kernel_size) for kernel_size in kernel_sizes])
        self.projection = nn.Sequential(
            nn.Linear(len(kernel_sizes) * hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, inputs: torch.Tensor):
        # Calculate and concatenate ngram features
        assert len(inputs.shape) == 3
        inputs = torch.transpose(inputs, 1, 2)  # Assume batch first
        conv_encodings = [conv(inputs) for conv in self.convolutions]  # Apply ngram encoder convolutions
        conv_encodings = torch.cat(conv_encodings, dim=1)  # Concatenate ngram features
        conv_encodings = torch.transpose(conv_encodings, 1, 2)

        # Project to common hidden dimension and apply relu
        encoded = self.projection(conv_encodings)

        return encoded
