import torch

from typing import Any
from torch import Tensor
from torch.optim import SGD
from constants import Sequence
from constants import SequenceData
from collections import defaultdict
from torch.optim import lr_scheduler
from torch_utils import move_to_cuda
from models.neural_plain_labeller import NeuralSequenceLabeller


class GramCTC(NeuralSequenceLabeller):
    def _calculate_loss(self, x_batch: Tensor, y_batch: SequenceData, lengths: Tensor, condition: Any = None):
        if self.cuda:
            x_batch = move_to_cuda(x_batch)
            condition = move_to_cuda(condition)

        batch_scores = self.model['encoder'](x_batch, lengths, condition)
        batch_scores = torch.log_softmax(batch_scores, dim=-1)

        batch_losses = []
        for scores, length, target in zip(batch_scores, lengths, y_batch):
            batch_losses.append(self._gram_ctc(scores=scores, target=target))

        loss = torch.stack(batch_losses).mean()
        return loss

    def _build_optimizer(self):
        return SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _build_lr_scheduler(self, dataset):
        return lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=len(dataset)
        )

    def _gram_ctc(self, scores: Tensor, target: Sequence):
        """
        Scores: Log prediction probabilities
        Target: Target string
        """

        # Helper function for ngram indices
        def get_ngram_idx(state):
            ngram = tuple(target[state[0] - state[1]:state[0]])
            idx = self.vocabulary.target2idx[ngram]
            return idx

        # Make states:
        # States are represented by the length of the prefix they represent and the length of the
        # last ngram
        states = []
        tau = max(len(ngram) for ngram in self.vocabulary.target2idx)

        for i in range(len(target) + 1):
            for j in range(1, min(tau, i) + 1):
                ngram = tuple(target[i - j:i])
                if ngram in self.vocabulary.target2idx:
                    states.append((i, j))

        # Define indices for states
        state2idx = {state: idx for idx, state in enumerate(states)}

        # Make state transition tree
        predecessors = defaultdict(list)

        for state, state_index in state2idx.items():
            for predecessor_candidate, predecessor_index in state2idx.items():
                ngram_idx = get_ngram_idx(state)
                blank_idx = self.vocabulary.target2idx[self.vocabulary.BLANK]

                if state[0] - state[1] == predecessor_candidate[0]:
                    predecessors[state_index].append((predecessor_index, ngram_idx))

                # Transition from another state with blank
                elif state_index == predecessor_index:
                    predecessors[state_index].append((state_index, blank_idx))

        # Start states have themselves as the only predecessor
        start_states = []
        for state in states:
            if state[0] == state[1]:
                start_states.append(state)

        # Final states end with the last index
        final_states = [state for state in states if state[0] == len(target)]

        if len(final_states) == 0:
            return torch.tensor(0.0, requires_grad=True).to(scores.device)

        # Calculate forward score matrix
        num_time_steps = scores.shape[0]  # Assume scores has shape time_steps x #classes
        num_states = len(states)

        alpha = [[torch.tensor(-1000) for _ in range(num_states)] for _ in range(num_time_steps)]

        # Initialise scores of start states
        for state in start_states:
            ngram_idx = get_ngram_idx(state)
            state_idx = state2idx[state]
            alpha[0][state_idx] = scores[0, ngram_idx]

        # Forward recursion
        for t in range(1, num_time_steps):
            for state, state_idx in state2idx.items():
                # Collect previous states and transition (= ngram) probabilities
                prev_scores = []
                for predecessor_idx, ngram_idx in predecessors[state_idx]:
                    prev_scores.append(alpha[t - 1][predecessor_idx] + scores[t][ngram_idx])
                prev_scores = torch.stack(prev_scores)
                alpha[t][state_idx] = torch.logsumexp(prev_scores, dim=0)

        # Partition function is sum of probabilities of final states
        final_state_probs = []
        for final_state in final_states:
            state_idx = state2idx[final_state]
            final_state_probs.append(alpha[-1][state_idx])

        final_state_probs = torch.stack(final_state_probs)
        target_probability = torch.logsumexp(final_state_probs, dim=0)

        # Return the negative log-likelihood
        return -target_probability
