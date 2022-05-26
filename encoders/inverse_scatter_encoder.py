import torch
import torch.nn as nn

from abc import ABC
from typing import Any
from encoders import Encoder
from encoders import LSTMEncoder
from encoders import NgramLSTMEncoder
from encoders import TransformerEncoder


class InverseScatterEncoder(Encoder, ABC):
    def __init__(self, *args, tau: int = 2, **kwargs):
        super(InverseScatterEncoder, self).__init__(*args, **kwargs)
        self.tau = tau

    def _initialise_classifier(self, num_classes: int):
        self.classifier = nn.Linear(self.hidden_size, self.tau * num_classes)

    def _get_prediction_scores(self, inputs: torch.Tensor, lengths: torch.Tensor, conditions: Any = None):
        scores = self.classifier(inputs)
        batch, timesteps, num_classes = scores.shape
        scores = torch.reshape(scores, shape=(batch, timesteps * self.tau, num_classes // self.tau))
        return scores


class InverseScatterLSTM(InverseScatterEncoder, LSTMEncoder):
    def get_save_info(self):
        info = super(InverseScatterLSTM, self).get_save_info()
        info['tau'] = self.tau
        return info


class InverseScatterTransformer(InverseScatterEncoder, TransformerEncoder):
    def get_save_info(self):
        info = super(InverseScatterTransformer, self).get_save_info()
        info['tau'] = self.tau
        return info


class InverseScatterNgramLSTM(InverseScatterEncoder, NgramLSTMEncoder):
    def get_save_info(self):
        info = super(InverseScatterNgramLSTM, self).get_save_info()
        info['tau'] = self.tau
        return info
