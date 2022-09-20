import torch
import torch.nn as nn

from abc import ABC
from torch import Tensor


class BaseCRF(nn.Module, ABC):
    def __init__(self, num_labels: int):
        super(BaseCRF, self).__init__()
        self.num_labels = num_labels

        self._prior = nn.Parameter(torch.zeros(num_labels))
        self._final_transition_scores = nn.Parameter(torch.zeros(num_labels))

    @property
    def prior(self) -> Tensor:
        return torch.log_softmax(self._prior, dim=-1)

    @property
    def final_transition_scores(self) -> Tensor:
        return torch.log_softmax(self._final_transition_scores, dim=-1)

    @property
    def transition_scores(self) -> Tensor:
        raise NotImplementedError

    def get_transition_scores(self, label_sequences: Tensor) -> Tensor:
        raise NotImplementedError


class CRF(BaseCRF):
    def __init__(self, num_labels: int):
        super(CRF, self).__init__(num_labels=num_labels)
        self._transition_scores = nn.Parameter(torch.zeros(num_labels, num_labels))

    @property
    def transition_scores(self, source_index: Tensor = None, target_index: Tensor = None) -> Tensor:
        if source_index is not None and target_index is not None:
            raise NotImplementedError

        return torch.log_softmax(self._transition_scores, dim=1)

    def get_transition_scores(self, label_sequences: Tensor) -> Tensor:
        return self.transition_scores[label_sequences[:, :-1], label_sequences[:, 1:]]


class TruncatedCRF(BaseCRF):
    def __init__(self, num_labels: int, embedding_dim: int):
        super(TruncatedCRF, self).__init__(num_labels=num_labels)
        self.embedding_dim = embedding_dim

        self._source_embeddings = nn.Parameter(torch.zeros(num_labels, embedding_dim))
        self._target_embeddings = nn.Parameter(torch.zeros(num_labels, embedding_dim))

    @property
    def transition_scores(self) -> Tensor:
        return torch.log_softmax(torch.mm(self._source_embeddings, self._target_embeddings.T), dim=1)

    def get_transition_scores(self, label_sequences: Tensor) -> Tensor:
        batch_size, timesteps = label_sequences.shape
        source_index = label_sequences[:, :-1].flatten()
        target_index = label_sequences[:, 1:].flatten()

        source_index_list = source_index.detach().cpu().tolist()
        source_index_set = list(set(source_index_list))
        source_index_mapping = {
            index: mapped_position for mapped_position, index in enumerate(source_index_set)
        }
        mapped_source_index = [source_index_mapping[index] for index in source_index_list]

        source_embeddings = self._source_embeddings[source_index_set]
        scores = torch.mm(source_embeddings, self._target_embeddings.T)
        scores = torch.log_softmax(scores, dim=1)
        scores = scores[mapped_source_index, target_index].contiguous()
        scores = scores.reshape(batch_size, timesteps - 1)
        return scores
