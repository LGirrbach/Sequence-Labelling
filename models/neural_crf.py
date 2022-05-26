import torch
import torch.nn as nn

from typing import Any
from typing import Dict
from typing import Tuple
from typing import List
from torch import Tensor
from constants import Sequence
from torch_utils import move_to_cuda
from torch_utils import make_mask_2d
from encoders.lstm_encoder import LSTMEncoder
from models.neural_labeller import NeuralLabeller

# This code uses some implementation ideas from
# https://github.com/jidasheng/bi-lstm-crf/blob/master/bi_lstm_crf/model/crf.py


class CRF(nn.Module):
    def __init__(self, num_tags: int, discriminative_loss: bool = False):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.discriminative_loss = discriminative_loss

        self.prior = nn.Parameter(torch.zeros(num_tags))
        self.transition_scores = nn.Parameter(torch.zeros(1, num_tags, num_tags))
        self.final_transition_scores = nn.Parameter(torch.zeros(num_tags))

    def forward(self, emission_scores: Tensor, lengths: Tensor, tags: Tensor) -> Tensor:
        if self.discriminative_loss:
            return self._discriminative_loss(emission_scores, lengths, tags)
        else:
            return self._nll_loss(emission_scores, lengths, tags)

    def _score_path(self, emission_scores: Tensor, lengths: Tensor, tags: Tensor) -> Tensor:
        # Get relevant dimension info
        batch, timesteps, num_tags = emission_scores.shape

        # Extract emission scores
        tag_index = tags.unsqueeze(2)
        emission_scores = torch.gather(emission_scores, index=tag_index, dim=2)
        emission_scores = emission_scores.squeeze(2)
        # Shape [Batch, Timesteps]

        # Extract transition scores
        transition_scores = self.transition_scores[0, tags[:, 1:], tags[:, :-1]]

        # Extract prior
        prior = self.prior[tags[:, 0]].contiguous()
        prior = prior.reshape(batch, 1)

        # Combine prior and transition scores
        transition_scores = torch.cat([prior, transition_scores], dim=1)

        # Calculate transition probabilities to stop tag
        length_index = (lengths - 1).unsqueeze(1).to(emission_scores.device)
        final_tags = torch.gather(tags, index=length_index, dim=1)
        final_tags = final_tags.reshape(batch)
        final_transition_scores = self.final_transition_scores[final_tags]
        final_transition_scores = final_transition_scores.contiguous()

        # Calculate path probabilities
        path_probabilities = transition_scores + emission_scores
        # Mask padding
        mask = make_mask_2d(lengths).to(emission_scores.device)
        path_probabilities = torch.masked_fill(path_probabilities, mask=mask, value=0.0)

        # Sum tag scores for each sequence
        path_probabilities = torch.sum(path_probabilities, dim=1)
        # Add transition probabilities to stop tag
        path_probabilities = path_probabilities + final_transition_scores

        return path_probabilities

    def _partition_function(self, emission_scores: Tensor, lengths: Tensor) -> Tensor:
        # Get relevant dimension info
        batch, timesteps, num_tags = emission_scores.shape

        # Check data integrity
        assert num_tags == self.num_tags

        # Normalise transition and emission scores
        transition_scores = self.transition_scores

        # Start with all paths have prior probability plus emission score
        prior = self.prior.expand(batch, num_tags)
        prev_alpha = emission_scores[:, 0, :].contiguous() + prior
        alpha = [prev_alpha]

        # Forward recursion
        for t in range(1, timesteps):
            emission_scores_t = emission_scores[:, t, :].unsqueeze(2)
            # Shape [Batch, 1, #Tags]
            prev_alpha = prev_alpha.unsqueeze(1)
            # Shape [Batch, #Tags, 1]

            alpha_t = prev_alpha + transition_scores + emission_scores_t
            # Shape [Batch, #Tags, #Tags]
            alpha_t = torch.logsumexp(alpha_t, dim=2)
            # Shape [Batch, #Tags]
            alpha.append(alpha_t)
            prev_alpha = alpha_t

        alpha = torch.stack(alpha)
        # Shape [Timesteps, Batch, #Tags]
        alpha = alpha.transpose(0, 1)
        # Shape [Batch, Timesteps, #Tags]

        # Get last scores for all batch elements
        alpha = alpha[torch.arange(batch), lengths-1]

        # Add final transition scores
        alpha = alpha + self.final_transition_scores.expand(batch, num_tags)

        # Calculate and return partition function
        return torch.logsumexp(alpha, dim=1)

    def _nll_loss(self, emission_scores: Tensor, lengths: Tensor, tags: Tensor) -> Tensor:
        # Assert data integrity
        lengths = torch.clamp(lengths, 1).long()

        # Calculate partition function scores
        partition_function = self._partition_function(emission_scores, lengths=lengths)
        # print(partition_function)

        # Calculate tag scores
        tag_scores = self._score_path(emission_scores, lengths, tags)
        # print(tag_scores)

        # Calculate negative-log-likelihood
        nll = partition_function - tag_scores
        return nll.mean()

    def _discriminative_loss(self, emission_scores: Tensor, lengths: Tensor, tags: Tensor) -> Tensor:
        # Assert data integrity
        lengths = torch.clamp(lengths, 1).long()

        # Calculate tag scores
        tag_scores = self._score_path(emission_scores, lengths, tags)

        # Calculate viterbi path scores
        viterbi_scores = self.viterbi_decode(emission_scores, lengths, return_paths=False)

        # Loss is difference of tag scores from viterbi scores
        loss = viterbi_scores - tag_scores
        return loss.mean()

    def viterbi_decode(self, emission_scores: Tensor, lengths: Tensor, return_paths: bool = True):
        # Get relevant dimension info
        batch, timesteps, num_tags = emission_scores.shape

        # Check data integrity
        assert num_tags == self.num_tags

        # Normalise transition and emission scores
        transition_scores = self.transition_scores[0]

        # Calculate prior
        prior = self.prior.expand(batch, num_tags)

        # Start with emission scores at first time step times prior
        prev_alpha = emission_scores[:, 0, :].contiguous()
        prev_alpha = prior + prev_alpha
        alpha = [prev_alpha]

        # Initialise back-pointers
        back_pointers = torch.zeros(batch, timesteps, num_tags).long()

        # Forward recursion
        for t in range(1, timesteps):
            emission_scores_t = emission_scores[:, t, :]
            # Shape [Batch, 1, #Tags]
            prev_alpha = prev_alpha.unsqueeze(1)
            # Shape [Batch, #Tags, 1]

            alpha_t = prev_alpha + transition_scores
            # Shape [Batch, #Tags, #Tags]
            # Get maximum values for each tag
            alpha_t, back_pointers_t = torch.max(alpha_t, dim=2)
            alpha_t = alpha_t + emission_scores_t
            alpha.append(alpha_t)
            back_pointers[:, t, :] = back_pointers_t
            prev_alpha = alpha_t

        alpha = torch.stack(alpha)
        alpha = alpha.transpose(0, 1)

        viterbi_scores = alpha[torch.arange(batch), lengths-1]
        viterbi_scores = viterbi_scores + self.final_transition_scores.expand(batch, num_tags)

        viterbi_scores, start_tag_indices = torch.max(viterbi_scores, dim=1)
        if not return_paths:
            return viterbi_scores

        # Reconstruct predicted paths from back-pointers
        start_tag_indices = start_tag_indices.tolist()
        back_pointers = back_pointers.detach().cpu().numpy()
        predicted_paths = []

        for batch_idx, length in enumerate(lengths.tolist()):
            if length == 0:
                predicted_paths.append([])
                continue

            start_idx = start_tag_indices[batch_idx]
            predicted_path = [start_idx]

            for t in range(1, length):
                start_idx = back_pointers[batch_idx, length-t, start_idx].item()
                predicted_path.append(start_idx)

            predicted_path = list(reversed(predicted_path))
            try:
                assert len(predicted_path) == length
            except AssertionError:
                print(predicted_path)
                print(length)
                raise

            predicted_paths.append(predicted_path)

        return predicted_paths, viterbi_scores


class NeuralCRF(NeuralLabeller):
    def __init__(self, *args, discriminative_loss: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminative_loss = discriminative_loss
        self.crf = None

    def get_params(self) -> Dict:
        params = super(NeuralCRF, self).get_params()
        params['discriminative_loss'] = self.discriminative_loss
        return params

    def build_model(self):
        # Initialise model and CRF layer
        self.model = LSTMEncoder() if self.model is None else self.model
        self.model.initialise(self.vocabulary.num_source(), self.vocabulary.num_target())
        self.crf = CRF(self.vocabulary.num_target(), discriminative_loss=self.discriminative_loss)

        self.model = nn.ModuleDict({'encoder': self.model, 'crf': self.crf})
        self.model = self.model.cuda() if self.cuda else self.model.cpu()

        return self.model

    def _calculate_loss(self, x_batch: Tensor, y_batch: Tensor, lengths: Tensor, condition: Tuple[Any, ...]):
        if self.cuda:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            condition = move_to_cuda(condition)

        emission_scores = self.model['encoder'](x_batch, lengths, condition)
        crf_loss = self.model['crf'](emission_scores, lengths, y_batch)

        return crf_loss

    def _decode_prediction(self, prediction_scores: Tensor, length: List[int]) -> Sequence:
        length = torch.LongTensor(length).cpu()
        predicted_indices, _ = self.model['crf'].viterbi_decode(prediction_scores, length)
        predicted_labels = [self.vocabulary.decode_target(prediction) for prediction in predicted_indices]
        return predicted_labels
