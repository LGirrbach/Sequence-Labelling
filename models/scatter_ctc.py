import torch

from typing import Any
from typing import List
from typing import Dict
from torch import Tensor
from torch.optim import SGD
from constants import Sequence
from torch_utils import move_to_cuda
from torch.optim import lr_scheduler
from torch.nn.functional import ctc_loss
from vocabulary import ScatterCTCVocabulary
from models.neural_plain_labeller import NeuralSequenceLabeller


class ScatterCTCLabeller(NeuralSequenceLabeller):
    """
    Implements a scattering CTC objective:
    To overcome problems with differing lengths of input and target sequences, we repeat input tokens a fixed
    number of times and use CTC loss to predict targets.
    """
    def _calculate_loss(self, x_batch: Tensor, y_batch: Tensor, lengths: Tensor, condition: Any = None) -> Tensor:
        if self.cuda:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            condition = move_to_cuda(condition)

        # Get prediction probabilities
        y_predicted = self.model['encoder'](x_batch, lengths, condition)
        # Apply log-softmax
        log_probs = torch.log_softmax(y_predicted, dim=-1)
        log_probs = torch.transpose(log_probs, 0, 1)
        targets = y_batch

        source_lengths = lengths
        target_lengths = (y_batch.cpu() != 0).sum(dim=1).flatten()

        # Calculate CTC loss
        assert isinstance(self.vocabulary, ScatterCTCVocabulary)
        loss = ctc_loss(
            log_probs=log_probs, targets=targets, input_lengths=source_lengths, target_lengths=target_lengths,
            blank=self.vocabulary.blank_idx
        )

        return loss

    def _build_optimizer(self):
        return SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _build_lr_scheduler(self, dataset):
        self.update_lr_every_step = True
        return lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=len(dataset)
        )

    def postprocess_prediction(self, prediction: Sequence) -> Sequence:
        reduced_labels = []
        previous_label = None

        for label in prediction:
            if label in self.vocabulary.SPECIALS:
                previous_label = None

            elif label != previous_label:
                previous_label = label
                reduced_labels.append(label)

        return reduced_labels


class InverseScatterCTCLabeller(ScatterCTCLabeller):
    def __init__(self, *args, tau: int = 2, **kwargs):
        super(InverseScatterCTCLabeller, self).__init__(*args, **kwargs)
        self.tau = tau

    def get_params(self) -> Dict:
        params = super(InverseScatterCTCLabeller, self).get_params()
        params['tau'] = self.tau
        return params

    def _calculate_loss(self, x_batch: Tensor, y_batch: Tensor, lengths: Tensor, condition: Any = None):
        if self.cuda:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            condition = move_to_cuda(condition)

        # Get prediction probabilities
        y_predicted = self.model['encoder'](x_batch, lengths, condition)
        # Apply log-softmax
        log_probs = torch.log_softmax(y_predicted, dim=-1)
        log_probs = torch.transpose(log_probs, 0, 1)
        targets = y_batch

        source_lengths = self.tau * lengths
        target_lengths = (y_batch.cpu() != 0).sum(dim=1).flatten()

        # Calculate CTC loss
        loss = ctc_loss(
            log_probs=log_probs, targets=targets, input_lengths=source_lengths, target_lengths=target_lengths,
            blank=0
        )

        return loss

    def _decode_prediction(self, prediction_scores: Tensor, length: List[int]) -> List[str]:
        """
        Predict labels from label prediction scores calculated by encoder.
        Default implementation is greedy decoding.
        """
        batch_predictions = []

        for prediction_scores, length in zip(prediction_scores, length):
            length = int(length) * self.tau
            # Truncate scores
            prediction_scores = prediction_scores[:length]
            # Take maximum likelihood predictions
            predicted_indices = torch.argmax(prediction_scores, dim=-1)
            predicted_indices = predicted_indices.detach().cpu().tolist()
            # Decode labels
            predicted_labels = self.vocabulary.decode_target(predicted_indices)
            batch_predictions.append(predicted_labels)

        return batch_predictions
