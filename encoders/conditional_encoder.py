import torch

from abc import ABC
from typing import Any
from typing import Tuple
from encoders import Encoder


class ConditionalEncoder(Encoder, ABC):
    """Base class for conditional sequence labeller"""
    def _initialise_condition_encoder(self):
        raise NotImplementedError

    def initialise(self, vocab_size: int, num_classes: int):
        super(ConditionalEncoder, self).initialise(vocab_size=vocab_size, num_classes=num_classes)
        self._initialise_condition_encoder()

    def _apply_condition_encoder(self, conditions: Tuple[Any, ...]):
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor, conditions: Tuple[Any, ...] = None):
        embedded = self._embed(inputs)
        conditions = self._apply_condition_encoder(conditions)
        encoded = self._apply_encoder(embedded, lengths, conditions)
        scores = self._get_prediction_scores(encoded, lengths, conditions)

        return scores
