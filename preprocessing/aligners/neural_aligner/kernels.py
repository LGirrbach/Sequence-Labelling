import torch
import torch.nn as nn


class Kernel(nn.Module):
    """
    Base class for similarity kernels.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def kernel_factory(kernel_type, **parameters):
        if kernel_type == 'linear':
            return LinearKernel()

        elif kernel_type == 'bilinear':
            return BiLinearKernel(**parameters)

        elif kernel_type == 'polynomial':
            return PolynomialKernel(**parameters)

        elif kernel_type == 'rbf':
            return RBFKernel(**parameters)

        elif kernel_type == 'euclidean':
            return EuclideanSimilarityKernel()

        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")


class LinearKernel(Kernel):
    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor):
        assert len(x.shape) == len(y.shape)
        num_dimensions = len(x.shape)

        if num_dimensions == 2:
            return torch.mm(x, y.T)

        elif num_dimensions == 3:
            return torch.bmm(x, y.transpose(1, 2))

        else:
            raise RuntimeError(f"Matrix must have 2 or 3 dimensions, but has {num_dimensions}")


class BiLinearKernel(Kernel):
    def __init__(self, num_features, **parameters):
        super().__init__()
        self.weights = nn.Linear(num_features, num_features)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.weights(x)
        return LinearKernel.forward(x, y)


class PolynomialKernel(Kernel):
    def __init__(self, degree: float, **parameters):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(0.0))
        self.degree = degree

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        similarities = LinearKernel.forward(x, y) + self.c
        similarities = torch.pow(similarities, self.degree)
        return similarities


class RBFKernel(Kernel):
    def __init__(self, length_scale: float = 1., **parameters):
        super().__init__()
        self.length_scale = length_scale

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x, y = x.contiguous(), y.contiguous()
        distances = torch.cdist(x, y)
        distances = distances / (2 * self.length_scale ** 2)
        return torch.exp(- distances)


class EuclideanSimilarityKernel(Kernel):
    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor):
        x, y = x.contiguous(), y.contiguous()
        distances = torch.cdist(x, y)
        return 1. / (1 + distances)
