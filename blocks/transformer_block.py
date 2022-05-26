import torch
import torch.nn as nn

from torch_utils import make_mask_2d


class TransformerBlock(nn.Module):
    """
    Implements a wrapper around pytorch's transformer implementation for easier sequence processing.
    Automatically calculates attention mask from given sequence lengths.
    Note: Projects outputs of transformer to given hidden size.
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1, dim_feedforward: int = 2048,
                 nhead: int = 4, dropout: float = 0.0):
        super(TransformerBlock, self).__init__()

        # Save arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.dropout = dropout

        # Make modules
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
        self.projection = nn.Linear(input_size, hidden_size)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
        mask = make_mask_2d(lengths)  # Make attention mask
        encoded = self.transformer(inputs, src_key_padding_mask=mask)  # Apply transformer
        encoded = self.projection(encoded)  # Project outputs to hidden dimension

        return encoded
