import torch
import torch.nn as nn

from abc import ABC
from typing import Tuple, Any, Dict


class Encoder(nn.Module, ABC):
    """
    Base class for (non-conditional) sequence labelling encoders.
    """
    def __init__(self, *args, **kwargs):
        super(Encoder, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def get_save_info(self):
        """Return information necessary to load persistent model"""
        raise NotImplementedError

    @classmethod
    def load(cls, save_info: Dict):
        """Instantiate model from saved info"""
        return cls(**save_info)

    def _initialise_embedding_layer(self, vocab_size: int):
        self.embedding = nn.Embedding(vocab_size, self.embedding_size, padding_idx=0)

    def _initialise_encoder(self):
        raise NotImplementedError

    def _initialise_classifier(self, num_classes: int):
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def initialise(self, vocab_size: int, num_classes: int):
        self._initialise_embedding_layer(vocab_size)
        self._initialise_encoder()
        self._initialise_classifier(num_classes)

    def _embed(self, inputs: torch.Tensor):
        return self.embedding(inputs)

    def _apply_encoder(self, inputs: torch.Tensor, lengths: torch.Tensor, conditions: Any = None):
        raise NotImplementedError

    def _get_prediction_scores(self, inputs: torch.Tensor, lengths: torch.Tensor, conditions: Any = None):
        return self.classifier(inputs)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor, conditions: Tuple[Any, ...] = None):
        embedded = self._embed(inputs)
        encoded = self._apply_encoder(embedded, lengths, conditions)
        scores = self._get_prediction_scores(encoded, lengths, conditions)

        return scores
