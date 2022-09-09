import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_labels: int):
        super(CRF, self).__init__()
        self.num_labels = num_labels

        self._prior = nn.Parameter(torch.zeros(num_labels))
        self._transition_scores = nn.Parameter(torch.zeros(num_labels, num_labels))
        self._final_transition_scores = nn.Parameter(torch.zeros(num_labels))

    @property
    def prior(self) -> torch.Tensor:
        return torch.log_softmax(self._prior, dim=-1)

    @property
    def transition_scores(self) -> torch.Tensor:
        return torch.log_softmax(self._transition_scores, dim=0)

    @property
    def final_transition_scores(self) -> torch.Tensor:
        return torch.log_softmax(self._final_transition_scores, dim=-1)
