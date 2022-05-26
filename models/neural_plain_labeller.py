import torch
import torch.nn as nn

from typing import Any
from torch import LongTensor
from torch import FloatTensor
from encoders.lstm_encoder import LSTMEncoder
from torch.nn.functional import cross_entropy
from models.neural_labeller import NeuralLabeller


class NeuralSequenceLabeller(NeuralLabeller):
    def build_model(self):
        # Initialise model
        self.model = LSTMEncoder() if self.model is None else self.model
        self.model.initialise(self.vocabulary.num_source(), self.vocabulary.num_target())
        self.model = nn.ModuleDict({'encoder': self.model})
        self.model = self.model.cuda() if self.cuda else self.model.cpu()

        return self.model

    def _calculate_loss(self, x_batch: FloatTensor, y_batch: LongTensor, lengths: LongTensor, condition: Any = None):
        if self.cuda:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            condition = [cond.cuda() for cond in condition]

        # Get prediction probabilities
        y_predicted = self.model['encoder'](x_batch, lengths, condition)
        # Flatten scores and labels (necessary for cross entropy loss)
        y_predicted = torch.flatten(y_predicted, end_dim=-2)
        y_true = torch.flatten(y_batch)
        # Calculate cross entropy loss
        loss = cross_entropy(y_predicted, y_true, ignore_index=0)

        return loss
