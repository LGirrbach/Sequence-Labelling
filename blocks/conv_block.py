import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Implements a 1d-convolutional layer consisting of convolution, relu activation, dropout, and (optionally) batch
    norm.
    """
    def __init__(self, input_size: int, hidden_size: int = 128, kernel_size: int = 3, dropout: float = 0.0,
                 batch_norm: bool = False):
        super(ConvLayer, self).__init__()

        # Save arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.batch_norm = batch_norm

        # Make modules
        conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding='same')
        relu = nn.ReLU()
        dropout = nn.Dropout(p=self.dropout)
        norm = nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity()

        self.conv = nn.Sequential(conv, relu, dropout, norm)

    def forward(self, inputs: torch.Tensor):
        return self.conv(inputs)


class ResidualConvLayer(nn.Module):
    """
    Implements a residual 1d-convolution layer.
    Transform consists of 1d-convolution, relu activation, dropout and (optionally) batch norm.
    Inputs are added after conv and relu, but before dropout and batch norm.
    """
    def __init__(self, input_size: int, kernel_size: int = 3, dropout: float = 0.0, batch_norm: bool = False):
        super(ResidualConvLayer, self).__init__()

        # Save arguments
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.batch_norm = batch_norm

        # Make modules
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size, padding='same'),
            nn.ReLU()
        )

        norm = nn.BatchNorm1d(num_features=input_size) if self.batch_norm else nn.Identity()
        self.dropout_and_norm = nn.Sequential(nn.Dropout(p=dropout), norm)

    def forward(self, inputs: torch.Tensor):
        transformed = self.conv(inputs)  # Transform inputs
        transformed = inputs + transformed  # Add inputs to transformed
        transformed = self.dropout_and_norm(transformed)

        return transformed


class ConvBlock(nn.Module):
    """
    Implements a 1d-convolution encoder consisting of multiple convolutional layers.
    Note: Different from the standard pytorch implementation, this assumes batch first.
    Note: Dropout and batch norm are only applied to intermediate layers, not last layer.
    Note: If input size != hidden size, the first layers is cannot be residual
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1, kernel_size: int = 3,
                 dropout: float = 0.0, batch_norm: bool = False, residual: bool = False):
        super(ConvBlock, self).__init__()

        # Save arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        # Make modules
        modules = []

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            layer_dropout = dropout if layer + 1 < num_layers else 0.0  # Only use dropout for intermediate layers
            layer_batch_norm = True if layer + 1 < num_layers else False  # Only use batch norm for intermediate layers

            # Use standard conv layer if not residual or input size is not hidden size
            if (layer == 0 and layer_input_size != hidden_size) or not self.residual:
                layer = ConvLayer(layer_input_size, hidden_size, kernel_size, layer_dropout, layer_batch_norm)
            else:
                layer = ResidualConvLayer(layer_input_size, kernel_size, layer_dropout, layer_batch_norm)

            modules.append(layer)

        self.conv = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor):
        encoded = torch.transpose(inputs, -1, -2)  # Assume batch first
        encoded = self.conv(encoded)
        encoded = torch.transpose(encoded, -1, -2)

        return encoded
